from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_analytics.data_generator import generate_all
from healthcare_analytics.llm_reporting import generate_executive_summary
from healthcare_analytics.metrics import build_metric_outputs
from healthcare_analytics.modeling import compare_forecasting_models, detect_reporting_anomalies, forecast_monthly_cases
from healthcare_analytics.pipeline import run_pipeline
from healthcare_analytics.rag import RetrievedContext, answer_question, build_rag_prompt, classify_query_scope, classify_source_type, load_chunks, looks_report_facility_question, rerank_contexts, split_markdown_into_chunks, write_metric_knowledge_documents
from healthcare_analytics.rag_evaluation import _groundedness_score, _keyword_score
from healthcare_analytics.validation import run_validation
from healthcare_analytics.weather_tool import extract_weather_location, looks_weather_question
from healthcare_analytics.local_llm import clean_generated_text


class PipelineTests(unittest.TestCase):
    def test_generate_validate_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            data = generate_all(raw_dir, records=500, months=8, seed=13)
            self.assertIn("trauma_registry", data)
            self.assertGreater(len(data["trauma_registry"]), 0)
            self.assertTrue((raw_dir / "trauma_registry.csv").exists())

            issues = run_validation(data)
            self.assertIsInstance(issues, pd.DataFrame)

            outputs = build_metric_outputs(data, issues)
            self.assertIn("executive_overview", outputs)
            self.assertGreater(len(outputs["executive_overview"]), 0)

            anomalies = detect_reporting_anomalies(outputs["facility_monthly_submissions"])
            forecast = forecast_monthly_cases(outputs["executive_overview"], horizon=3)
            comparison = compare_forecasting_models(outputs["executive_overview"])
            self.assertEqual(len(forecast), 3)
            self.assertIn("linear_trend_with_seasonal_anchor", set(comparison["model"]))

            summary = generate_executive_summary(outputs, anomalies=anomalies, forecasts=forecast)
            self.assertIn("Executive Summary", summary)
            self.assertIn("synthetic data only", summary.lower())

    def test_run_pipeline_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = run_pipeline(project_root=tmp, records=400, months=6, seed=21)
            self.assertTrue(paths["executive_summary"].exists())
            self.assertTrue(paths["rag_metric_snapshot"].exists())
            self.assertTrue((Path(tmp) / "data" / "processed" / "executive_overview.csv").exists())
            self.assertTrue((Path(tmp) / "data" / "processed" / "case_volume_forecast.csv").exists())
            self.assertTrue((Path(tmp) / "data" / "processed" / "forecast_model_comparison.csv").exists())

    def test_rag_chunking_and_prompt(self) -> None:
        text = "# Metric Definitions\n\nFatality rate is fatalities divided by total trauma cases.\n\n## Governance\n\nUse aggregate synthetic data only."
        chunks = split_markdown_into_chunks(text, source="knowledge_base/test.md", chunk_size=90, overlap=10)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("Fatality rate", " ".join(chunk.text for chunk in chunks))

        prompt = build_rag_prompt(
            "How is fatality rate calculated?",
            [
                RetrievedContext(
                    text=chunks[0].text,
                    source=chunks[0].source,
                    section=chunks[0].section,
                    score=0.1,
                )
            ],
        )
        self.assertIn("provided context", prompt)
        self.assertIn("How is fatality rate calculated?", prompt)

    def test_metric_snapshot_and_project_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_pipeline(project_root=root, records=300, months=6, seed=31)
            docs_dir = root / "knowledge_base"
            docs_dir.mkdir(exist_ok=True)
            (docs_dir / "sample.md").write_text("# Sample\n\nSynthetic RAG document.", encoding="utf-8")

            generated = write_metric_knowledge_documents(root)
            self.assertTrue(generated[0].exists())
            chunks = load_chunks(root)
            self.assertGreater(len(chunks), 0)
            self.assertTrue(any("Latest Aggregate Metric Snapshot" in chunk.text for chunk in chunks))

    def test_greeting_uses_hybrid_chat_mode_without_rag(self) -> None:
        result = answer_question("Hi", project_root=PROJECT_ROOT)
        self.assertEqual(result.mode, "greeting")
        self.assertIn("dashboard", result.answer.lower())
        self.assertEqual(result.sources, [])

    def test_weather_question_location_detection(self) -> None:
        question = "How is weather today in Albany, NY?"
        self.assertTrue(looks_weather_question(question))
        self.assertEqual(extract_weather_location(question), "Albany, NY")

    def test_llm_output_preserves_markdown_structure(self) -> None:
        text = "Heading\n\n- first item\n- second item"
        self.assertEqual(clean_generated_text(text), text)

    def test_report_facility_questions_prefer_report_sources(self) -> None:
        question = "Give a summary of audit report for each facility"
        self.assertTrue(looks_report_facility_question(question))
        self.assertEqual(classify_query_scope(question), "internal")
        self.assertEqual(classify_source_type("knowledge_base/generated/audit_report_facility_summary.md"), "report_structured")
        contexts = [
            RetrievedContext(
                text="Generated validation layer dashboard content.",
                source="knowledge_base/generated/latest_metric_snapshot.md",
                section="Data Quality",
                score=0.1,
                source_type="generated_dashboard",
            ),
            RetrievedContext(
                text="Audit report facility table with adult, pediatric, and total counts.",
                source="knowledge_base/generated/audit_report_facility_summary.md",
                section="Facility Summary Table",
                score=0.2,
                source_type="report_structured",
            ),
        ]
        selected = rerank_contexts(question, contexts, top_k=1)
        self.assertEqual(selected[0].source_type, "report_structured")

    def test_query_scope_classifier_keeps_external_queries_out_of_rag(self) -> None:
        self.assertEqual(classify_query_scope("How is weather today in Albany, NY?"), "weather")
        self.assertEqual(classify_query_scope("What is the latest CDC flu guidance?"), "external")
        self.assertEqual(classify_query_scope("What are the main trauma metrics?"), "internal")
        self.assertEqual(classify_query_scope("Write a short Python list comprehension example"), "general")

    def test_rag_evaluation_scores(self) -> None:
        answer = "The RAG workflow uses embeddings, Chroma retrieval, chunks, and Ollama."
        self.assertGreaterEqual(_keyword_score(answer, ["embedding", "chroma", "ollama"]), 0.66)
        self.assertGreater(_groundedness_score(answer, "Chroma retrieval uses embeddings and chunks before Ollama answers."), 0)


if __name__ == "__main__":
    unittest.main()
