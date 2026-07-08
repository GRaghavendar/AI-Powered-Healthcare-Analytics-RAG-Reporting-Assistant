"""End-to-end command line pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import DEFAULT_END_MONTH, DEFAULT_MONTHS, DEFAULT_RECORDS, PROJECT_ROOT
from .data_generator import generate_all
from .llm_reporting import generate_executive_summary, generate_facility_narratives
from .metrics import build_metric_outputs, save_metric_outputs
from .modeling import compare_forecasting_models, detect_reporting_anomalies, forecast_monthly_cases
from .rag import write_metric_knowledge_documents
from .validation import run_validation

RAW_FILE_NAMES = {
    "hospital_master": "hospital_master.csv",
    "population_reference": "population_reference.csv",
    "trauma_registry": "trauma_registry.csv",
    "ems_records": "ems_records.csv",
    "hospital_transfer": "hospital_transfer.csv",
    "claims_data": "claims_data.csv",
    "organ_donor_registry": "organ_donor_registry.csv",
}


def load_raw_data(raw_dir: str | Path) -> Dict[str, pd.DataFrame]:
    raw_path = Path(raw_dir)
    data = {}
    for name, filename in RAW_FILE_NAMES.items():
        path = raw_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw dataset: {path}")
        data[name] = pd.read_csv(path)
    return data


def run_pipeline(
    project_root: str | Path = PROJECT_ROOT,
    records: int = DEFAULT_RECORDS,
    months: int = DEFAULT_MONTHS,
    seed: int = 42,
    end_month: str = DEFAULT_END_MONTH,
    generate_data: bool = True,
    build_rag_index: bool = False,
    rag_ocr: bool = False,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if generate_data:
        data = generate_all(raw_dir, records=records, months=months, seed=seed, end_month=end_month)
    else:
        data = load_raw_data(raw_dir)

    validation_issues = run_validation(data)
    metric_outputs = build_metric_outputs(data, validation_issues)
    anomalies = detect_reporting_anomalies(metric_outputs["facility_monthly_submissions"])
    forecasts = forecast_monthly_cases(metric_outputs["executive_overview"], horizon=3)
    forecast_comparison = compare_forecasting_models(metric_outputs["executive_overview"])

    metric_outputs["reporting_anomalies"] = anomalies
    metric_outputs["case_volume_forecast"] = forecasts
    metric_outputs["forecast_model_comparison"] = forecast_comparison
    save_metric_outputs(metric_outputs, processed_dir)

    executive_summary = generate_executive_summary(metric_outputs, anomalies=anomalies, forecasts=forecasts)
    facility_narratives = generate_facility_narratives(metric_outputs["hospital_comparison"], anomalies=anomalies)
    (reports_dir / "executive_summary.md").write_text(executive_summary, encoding="utf-8")
    (reports_dir / "facility_narratives.md").write_text(facility_narratives, encoding="utf-8")
    write_metric_knowledge_documents(root)

    if build_rag_index:
        from .rag import build_chroma_index

        build_chroma_index(project_root=root, reset=True, use_ocr=rag_ocr)

    outputs = {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "reports_dir": reports_dir,
        "executive_summary": reports_dir / "executive_summary.md",
        "facility_narratives": reports_dir / "facility_narratives.md",
        "rag_metric_snapshot": root / "knowledge_base" / "generated" / "latest_metric_snapshot.md",
        "forecast_model_comparison": processed_dir / "forecast_model_comparison.csv",
    }
    if build_rag_index:
        outputs["chroma_vector_dir"] = root / "vector_store" / "chroma"
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the healthcare synthetic trauma analytics pipeline.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root containing data/ and reports/.")
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS, help="Synthetic trauma records to generate.")
    parser.add_argument("--months", type=int, default=DEFAULT_MONTHS, help="Number of months to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--end-month", default=DEFAULT_END_MONTH, help="Final month in YYYY-MM-DD or YYYY-MM format.")
    parser.add_argument("--skip-data-generation", action="store_true", help="Use existing CSV files under data/raw.")
    parser.add_argument("--build-rag-index", action="store_true", help="Build the Chroma RAG index after creating metrics.")
    parser.add_argument("--rag-ocr", action="store_true", help="Use OCR while building the RAG index.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = run_pipeline(
        project_root=args.project_root,
        records=args.records,
        months=args.months,
        seed=args.seed,
        end_month=args.end_month,
        generate_data=not args.skip_data_generation,
        build_rag_index=args.build_rag_index,
        rag_ocr=args.rag_ocr,
    )
    print("Pipeline complete.")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
