"""RAG pipeline using SentenceTransformers embeddings and Chroma."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import re

import pandas as pd

from .config import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    KNOWLEDGE_BASE_DIR,
    PROMPTS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    VECTOR_DB_DIR,
)
from .document_ingestion import extract_knowledge_documents
from .local_llm import LocalLLM, LocalLLMConfig
from .web_search import WebSearchResult, search_duckduckgo
from .weather_tool import (
    WeatherResult,
    extract_weather_location,
    fetch_current_weather,
    looks_weather_question,
    weather_result_to_context,
)


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    section: str
    source_type: str = "knowledge_base"


@dataclass
class RetrievedContext:
    text: str
    source: str
    section: str
    score: float
    source_type: str = "knowledge_base"


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[RetrievedContext]
    prompt: str
    model_name: str
    embedding_model: str
    mode: str = "rag"
    web_results: list[WebSearchResult] | None = None
    weather_result: WeatherResult | None = None
    trace: list[str] = field(default_factory=list)


def _require_rag_dependencies():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "RAG dependencies are not installed. Run: pip install -r requirements.txt"
        ) from exc
    return chromadb, SentenceTransformer


def _chunk_source_id(source: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", source).strip("_")


def classify_source_type(source: str) -> str:
    normalized = source.replace("\\", "/").lower()
    if normalized.endswith("audit_report_facility_summary.md") or normalized.endswith("trauma_report_facility_reference.md"):
        return "report_structured"
    if "/extracted_documents/" in normalized or ".pdf.extracted.md" in normalized or normalized.endswith(".ocr.md"):
        return "report_pdf"
    if normalized.endswith("latest_metric_snapshot.md"):
        return "generated_dashboard"
    if normalized.startswith("reports/"):
        return "generated_report"
    if normalized.startswith("docs/"):
        return "project_docs"
    if normalized.startswith("knowledge_base/"):
        return "knowledge_base"
    return "unknown"


def split_markdown_into_chunks(text: str, source: str, chunk_size: int = 1100, overlap: int = 180) -> list[DocumentChunk]:
    """Split markdown into overlapping paragraph-aware chunks."""

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[DocumentChunk] = []
    current: list[str] = []
    current_size = 0
    section = "General"

    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            section = paragraph.lstrip("#").strip() or section
        paragraph_size = len(paragraph)
        if current and current_size + paragraph_size > chunk_size:
            chunk_text = "\n\n".join(current).strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{_chunk_source_id(source)}-{len(chunks):04d}",
                    text=chunk_text,
                    source=source,
                    section=section,
                    source_type=classify_source_type(source),
                )
            )
            tail = chunk_text[-overlap:] if overlap > 0 else ""
            current = [tail, paragraph] if tail else [paragraph]
            current_size = len(tail) + paragraph_size
        else:
            current.append(paragraph)
            current_size += paragraph_size

    if current:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{_chunk_source_id(source)}-{len(chunks):04d}",
                text="\n\n".join(current).strip(),
                source=source,
                section=section,
                source_type=classify_source_type(source),
            )
        )
    return chunks


def markdown_files(project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    root = Path(project_root)
    folders = [
        root / "knowledge_base",
        root / "docs",
        root / "reports",
    ]
    files: list[Path] = []
    for folder in folders:
        if folder.exists():
            files.extend(sorted(folder.rglob("*.md")))
    return files


def load_chunks(project_root: str | Path = PROJECT_ROOT) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    root = Path(project_root)
    for path in markdown_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        chunks.extend(split_markdown_into_chunks(text, source=rel))
    return chunks


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_metric_knowledge_documents(project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    """Convert aggregate processed outputs into markdown documents for RAG."""

    root = Path(project_root)
    processed_dir = root / "data" / "processed"
    generated_dir = root / "knowledge_base" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    overview = _safe_read_csv(processed_dir / "executive_overview.csv")
    hospital = _safe_read_csv(processed_dir / "hospital_comparison.csv")
    forecast = _safe_read_csv(processed_dir / "case_volume_forecast.csv")
    forecast_comparison = _safe_read_csv(processed_dir / "forecast_model_comparison.csv")
    anomalies = _safe_read_csv(processed_dir / "reporting_anomalies.csv")
    dq = _safe_read_csv(processed_dir / "data_quality_summary.csv")

    metric_path = generated_dir / "latest_metric_snapshot.md"
    lines = [
        "# Latest Aggregate Metric Snapshot",
        "",
        "This document is generated from synthetic aggregate CSV outputs. It contains no patient-level PHI.",
        "",
    ]
    if not overview.empty:
        latest = overview.sort_values("month").iloc[-1]
        lines.extend(
            [
                "## Current Reporting Month",
                f"- Month: {latest['month']}",
                f"- Total trauma cases: {int(latest['total_cases'])}",
                f"- Adult cases: {int(latest['adult_cases'])}",
                f"- Pediatric cases: {int(latest['pediatric_cases'])}",
                f"- Fatality rate: {latest['fatality_rate']:.2f}%",
                f"- ICU utilization rate: {latest['icu_rate']:.2f}%",
                f"- Ventilator utilization rate: {latest['ventilator_rate']:.2f}%",
                f"- Average hospital length of stay: {latest['avg_hospital_los_days']:.2f} days",
                f"- Claims paid amount: ${latest['total_paid_amount']:,.0f}",
                "",
            ]
        )
    if not hospital.empty:
        lines.append("## Hospital Comparison")
        for row in hospital.sort_values("total_cases", ascending=False).head(8).itertuples(index=False):
            lines.append(
                f"- {row.hospital_name}: {int(row.total_cases)} cases, "
                f"{row.fatality_rate:.2f}% fatality rate, {row.icu_rate:.2f}% ICU rate, "
                f"{row.avg_hospital_los_days:.2f} average LOS days."
            )
        lines.append("")
    if not forecast.empty:
        lines.append("## Case Volume Forecast")
        for row in forecast.itertuples(index=False):
            lines.append(
                f"- {row.month}: forecast {int(row.forecast_total_cases)} cases "
                f"with range {int(row.lower_bound)} to {int(row.upper_bound)}."
            )
        lines.append("")
    if not forecast_comparison.empty:
        lines.append("## Forecast Model Comparison")
        lines.append(
            "Baseline, trend, and optional machine-learning forecast models are compared with backtesting metrics. "
            "The transparent trend model remains the default because interpretability matters for reporting."
        )
        for row in forecast_comparison.itertuples(index=False):
            mae = "not run" if pd.isna(row.mae) else f"{row.mae:.2f}"
            rmse = "not run" if pd.isna(row.rmse) else f"{row.rmse:.2f}"
            mape = "not run" if pd.isna(row.mape) else f"{row.mape:.2f}%"
            default_note = " Default reporting model." if bool(row.selected_default) else ""
            lines.append(f"- {row.model}: MAE {mae}, RMSE {rmse}, MAPE {mape}.{default_note}")
        lines.append("")
    if not anomalies.empty and "anomaly_flag" in anomalies.columns:
        flagged = anomalies.loc[anomalies["anomaly_flag"].astype(str).str.lower().eq("true")].head(10)
        lines.append("## Reporting Anomalies")
        if flagged.empty:
            lines.append("- No current reporting anomalies are flagged.")
        else:
            for row in flagged.itertuples(index=False):
                lines.append(f"- {row.hospital_name} in {row.month}: {row.anomaly_reason}.")
        lines.append("")
    if not dq.empty:
        lines.append("## Data Quality Summary")
        for row in dq.itertuples(index=False):
            lines.append(
                f"- Dataset {row.dataset}, severity {row.severity}: "
                f"{int(row.issue_count)} issue types, {int(row.rows_affected)} rows affected."
            )
        lines.append("")
    metric_path.write_text("\n".join(lines), encoding="utf-8")

    return [metric_path]


def build_chroma_index(
    project_root: str | Path = PROJECT_ROOT,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    reset: bool = True,
    use_ocr: bool = False,
) -> dict:
    """Build or refresh the Chroma vector store from knowledge-base documents."""

    chromadb, SentenceTransformer = _require_rag_dependencies()
    root = Path(project_root).resolve()
    vector_dir = root / "vector_store" / "chroma"
    vector_dir.mkdir(parents=True, exist_ok=True)

    write_metric_knowledge_documents(root)
    ingestion_report = extract_knowledge_documents(root, use_ocr=use_ocr)
    chunks = load_chunks(root)
    if not chunks:
        raise RuntimeError("No markdown documents found to index.")

    client = chromadb.PersistentClient(path=str(vector_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)

    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode([chunk.text for chunk in chunks], normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "source": chunk.source,
                "section": chunk.section,
                "source_type": chunk.source_type,
                "embedding_model": embedding_model_name,
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
    )
    return {
        "collection_name": collection_name,
        "embedding_model": embedding_model_name,
        "chunk_count": len(chunks),
        "vector_dir": vector_dir,
        "pdf_files": ingestion_report.pdf_files,
        "image_files": ingestion_report.image_files,
        "pages_extracted": ingestion_report.pages_extracted,
        "ocr_pages": ingestion_report.ocr_pages,
        "ingestion_warnings": ingestion_report.warnings,
    }


def retrieve_context(
    question: str,
    project_root: str | Path = PROJECT_ROOT,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
) -> list[RetrievedContext]:
    chromadb, SentenceTransformer = _require_rag_dependencies()
    root = Path(project_root).resolve()
    direct_contexts = _load_direct_report_contexts(question, root, top_k)
    vector_dir = root / "vector_store" / "chroma"
    client = chromadb.PersistentClient(path=str(vector_dir))
    collection = client.get_collection(collection_name)
    model = SentenceTransformer(embedding_model_name)
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()[0]
    retrieval_k = max(top_k * 4, top_k)
    results = collection.query(query_embeddings=[query_embedding], n_results=retrieval_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    contexts: list[RetrievedContext] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        contexts.append(
            RetrievedContext(
                text=document,
                source=metadata.get("source", "unknown"),
                section=metadata.get("section", "unknown"),
                score=float(distance),
                source_type=metadata.get("source_type") or classify_source_type(metadata.get("source", "unknown")),
            )
        )
    return rerank_contexts(question, _dedupe_contexts([*direct_contexts, *contexts]), top_k)


PROJECT_TERMS = {
    "dashboard",
    "trauma",
    "fatality",
    "icu",
    "ventilator",
    "ems",
    "ed",
    "claims",
    "hospital",
    "Healthcare",
    "health research",
    "phi",
    "rag",
    "chroma",
    "embedding",
    "metric",
    "validation",
    "pipeline",
    "workflow",
    "anomaly",
    "forecast",
    "synthetic",
    "streamlit",
    "project",
}

KNOWLEDGE_BASE_DICTIONARY = {
    "analytics_outputs": {
        "dashboard",
        "metric",
        "metrics",
        "forecast",
        "anomaly",
        "validation",
        "data quality",
        "facility submissions",
        "executive overview",
        "hospital comparison",
    },
    "trauma_domain": {
        "trauma",
        "trauma registry",
        "trauma center",
        "fatality",
        "icu",
        "ventilator",
        "ems",
        "ed",
        "claims",
        "adult pediatric",
    },
    "reports_and_documents": {
        "audit report",
        "submitted cases",
        "submission report",
        "facility summary",
        "hospital list",
        "2016-2020",
        "trauma report",
        "data dictionary",
        "pdf report",
    },
    "system_design": {
        "rag",
        "chroma",
        "embedding",
        "vector database",
        "ollama",
        "langgraph",
        "streamlit",
        "pipeline",
        "workflow",
        "knowledge base",
        "phi",
        "synthetic data",
    },
}

REPORT_TERMS = {
    "audit report",
    "audit",
    "pdf",
    "document",
    "source document",
    "report",
    "submitted",
    "submission report",
    "facility",
    "facilities",
    "trauma center",
    "hospital list",
}

GREETING_TERMS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
WEB_TERMS = {"latest", "today", "current", "news", "internet", "web", "online", "recent"}
OUTSIDE_TERMS = {"public", "website", "search", "google", "duckduckgo", "official", "external"}
EXTERNAL_FACT_TERMS = {
    "president",
    "weather",
    "temperature",
    "stock",
    "market",
    "news",
    "population",
    "election",
    "price",
    "exchange rate",
    "release date",
}
DETAILED_TERMS = {
    "detail",
    "detailed",
    "deep",
    "explain",
    "describe",
    "step",
    "steps",
    "breakdown",
    "comprehensive",
    "complete",
    "full",
    "thorough",
}


def _normalized_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def is_greeting(question: str) -> bool:
    normalized = _normalized_question(question).strip(" .!?")
    return normalized in GREETING_TERMS or normalized in {"hi there", "hello there", "hey there"}


def looks_project_related(question: str) -> bool:
    normalized = _normalized_question(question)
    return any(term in normalized for term in PROJECT_TERMS)


def knowledge_base_matches(question: str) -> list[str]:
    normalized = _normalized_question(question)
    matches: list[str] = []
    for category, terms in KNOWLEDGE_BASE_DICTIONARY.items():
        if any(term in normalized for term in terms):
            matches.append(category)
    return matches


def classify_query_scope(question: str) -> str:
    """Classify the query before deciding whether Chroma should be used."""

    normalized = _normalized_question(question)
    if is_greeting(question):
        return "greeting"
    if looks_weather_question(question):
        return "weather"

    kb_matches = knowledge_base_matches(question)
    if kb_matches:
        return "internal"

    current_or_web = any(term in normalized for term in WEB_TERMS | OUTSIDE_TERMS | EXTERNAL_FACT_TERMS)
    factual_question = normalized.startswith(("who ", "who is ", "what is ", "what are ", "when ", "where ", "why ", "how many "))
    if current_or_web or factual_question:
        return "external"
    return "general"


def looks_report_facility_question(question: str) -> bool:
    normalized = _normalized_question(question)
    has_report_signal = any(term in normalized for term in REPORT_TERMS)
    has_facility_signal = any(term in normalized for term in ["facility", "facilities", "hospital", "hospitals", "trauma center"])
    asks_each = any(term in normalized for term in ["each", "list", "all", "summary"])
    return has_report_signal and (has_facility_signal or asks_each)


def _preferred_report_reference_sources(question: str) -> list[str]:
    normalized = _normalized_question(question)
    sources: list[str] = []
    if "audit" in normalized or "submitted" in normalized or "submission" in normalized:
        sources.append("knowledge_base/generated/audit_report_facility_summary.md")
    if "2016" in normalized or "2020" in normalized or "designation" in normalized or "case load" in normalized:
        sources.append("knowledge_base/generated/trauma_report_facility_reference.md")
    if not sources and looks_report_facility_question(question):
        sources.extend(
            [
                "knowledge_base/generated/audit_report_facility_summary.md",
                "knowledge_base/generated/trauma_report_facility_reference.md",
            ]
        )
    return sources


def _load_direct_report_contexts(question: str, project_root: str | Path, top_k: int) -> list[RetrievedContext]:
    root = Path(project_root)
    contexts: list[RetrievedContext] = []
    if not looks_report_facility_question(question):
        return contexts
    for source in _preferred_report_reference_sources(question):
        path = root / source
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        chunks = split_markdown_into_chunks(text, source=source, chunk_size=2400, overlap=120)
        for idx, chunk in enumerate(chunks[: max(top_k, 3)]):
            contexts.append(
                RetrievedContext(
                    text=chunk.text,
                    source=chunk.source,
                    section=chunk.section,
                    score=0.0 + (idx * 0.01),
                    source_type=chunk.source_type,
                )
            )
    return contexts[: max(top_k, 5)]


def _dedupe_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[RetrievedContext] = []
    for context in contexts:
        key = (context.source, context.section, context.text[:80])
        if key in seen:
            continue
        seen.add(key)
        unique.append(context)
    return unique


def rerank_contexts(question: str, contexts: list[RetrievedContext], top_k: int) -> list[RetrievedContext]:
    if not contexts:
        return []
    if looks_report_facility_question(question):
        preferred = [ctx for ctx in contexts if ctx.source_type in {"report_structured", "report_pdf"}]
        if preferred:
            contexts = preferred
    return sorted(contexts, key=lambda ctx: (ctx.score, 0 if ctx.source_type == "report_structured" else 1))[:top_k]


def looks_web_or_current(question: str) -> bool:
    normalized = _normalized_question(question)
    return looks_weather_question(question) or any(term in normalized for term in WEB_TERMS | OUTSIDE_TERMS)


def wants_detailed_answer(question: str) -> bool:
    normalized = _normalized_question(question)
    return any(term in normalized for term in DETAILED_TERMS)


def has_useful_rag_context(
    question: str,
    contexts: list[RetrievedContext],
    max_distance: float = 1.55,
) -> bool:
    if not contexts:
        return False
    if looks_project_related(question):
        return True
    return min(context.score for context in contexts) <= max_distance


def evidence_snippet(context: RetrievedContext, max_chars: int = 650) -> str:
    text = " ".join(context.text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_system_prompt(project_root: str | Path = PROJECT_ROOT) -> str:
    root = Path(project_root)
    prompt_path = root / "prompts" / "assistant_system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return """You are a helpful local healthcare analytics assistant. Answer clearly, use RAG evidence when relevant, use web snippets only when provided, and never claim synthetic data is real PHI."""


def build_rag_prompt(question: str, contexts: Iterable[RetrievedContext]) -> str:
    context_blocks = []
    for idx, context in enumerate(contexts, start=1):
        context_blocks.append(
            f"[Evidence {idx}: {context.section}]\n{context.text}"
        )
    context_text = "\n\n".join(context_blocks)
    return f"""Use the provided context to answer the user's question in clear, natural language.
Do not just list document paths. Explain the answer as if speaking to a dashboard user.

Provided context:
{context_text}

Question:
{question}

Answer:
"""


def build_general_prompt(question: str) -> str:
    return f"""Question:
{question}

No local RAG evidence was selected for this question. Answer as a normal helpful chatbot.

Answer:
"""


def build_web_prompt(question: str, web_results: list[WebSearchResult]) -> str:
    blocks = []
    for idx, result in enumerate(web_results, start=1):
        blocks.append(f"[Web result {idx}: {result.title}]\n{result.snippet}\nURL: {result.url}")
    web_context = "\n\n".join(blocks)
    return f"""Use the web search snippets below when they are relevant, and clearly say when the answer is based on web search.

Web search snippets:
{web_context}

Question:
{question}

Answer:
"""


def build_hybrid_prompt(
    question: str,
    contexts: list[RetrievedContext],
    web_results: list[WebSearchResult] | None = None,
    weather_result: WeatherResult | None = None,
    mode: str = "rag",
) -> str:
    context_blocks = []
    for idx, context in enumerate(contexts, start=1):
        label = context.source_type.replace("_", " ").title()
        context_blocks.append(
            f"[{label} evidence {idx}: {context.section} | {context.source}]\n{context.text}"
        )
    web_blocks = []
    for idx, result in enumerate(web_results or [], start=1):
        web_blocks.append(f"[Web evidence {idx}: {result.title}]\n{result.snippet}\nURL: {result.url}")
    weather_context = weather_result_to_context(weather_result) if weather_result else "No live weather context retrieved."

    rag_context = "\n\n".join(context_blocks) or "No local project/report context retrieved."
    web_context = "\n\n".join(web_blocks) or "No web context retrieved."
    detail_instruction = (
        "The user asked for a detailed explanation. Use markdown headings and bullets. "
        "Include: overview, purpose, data/submission flow, key fields or metrics, quality/audit checks, "
        "how the report is used, limitations, and a brief source note."
        if wants_detailed_answer(question)
        else "Answer at a normal length. Use bullets only if they improve clarity."
    )
    return f"""Assistant mode: {mode}

Local RAG evidence:
{rag_context}

Web evidence:
{web_context}

Live weather evidence:
{weather_context}

Instructions for this answer:
- Give a direct, useful chatbot answer first.
- {detail_instruction}
- Use PDF/report evidence for audit-report, facility-list, submitted-cases, and source-document questions.
- Use generated dashboard evidence only for dashboard metrics, synthetic analytics outputs, validation checks, and pipeline questions.
- Do not mix PDF report facts with generated synthetic dashboard metrics unless the user explicitly asks for that comparison.
- Use live weather evidence for current weather questions.
- Use web evidence only if it is relevant.
- If the local evidence does not answer the question, say so briefly and then answer with general knowledge if safe.
- Do not only return file names, page numbers, or source paths.
- Preserve paragraph breaks and markdown formatting.

User question:
{question}

Answer:
"""


def _answer_question_manual(
    question: str,
    project_root: str | Path = PROJECT_ROOT,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model_name: str = DEFAULT_LLM_MODEL,
    top_k: int = 5,
    llm: LocalLLM | None = None,
    use_web: bool = True,
    use_weather: bool = True,
) -> RagAnswer:
    trace: list[str] = ["start: received user question"]
    query_scope = classify_query_scope(question)
    trace.append(f"classify_question: scope={query_scope}")
    if query_scope == "greeting":
        prompt = build_general_prompt(question)
        answer = (
            "Hi! I can help you understand the trauma analytics dashboard, explain the RAG workflow, "
            "summarize metrics, review data quality checks, or answer general questions."
        )
        return RagAnswer(
            question=question,
            answer=answer,
            sources=[],
            prompt=prompt,
            model_name=llm_model_name,
            embedding_model=embedding_model_name,
            mode="greeting",
            web_results=[],
            weather_result=None,
            trace=["classify_question: greeting", "generate_answer: returned lightweight greeting"],
        )

    generator = llm or LocalLLM(LocalLLMConfig(model_name=llm_model_name))
    system_prompt = load_system_prompt(project_root)

    contexts: list[RetrievedContext] = []
    use_rag = False
    if query_scope == "internal":
        try:
            contexts = retrieve_context(
                question=question,
                project_root=project_root,
                collection_name=collection_name,
                embedding_model_name=embedding_model_name,
                top_k=top_k,
            )
            source_types = sorted({context.source_type for context in contexts})
            trace.append(f"retrieve_rag: retrieved {len(contexts)} chunks; source_types={source_types}")
        except Exception:
            contexts = []
            trace.append("retrieve_rag: Chroma index unavailable or empty")
        use_rag = has_useful_rag_context(question, contexts)
        trace.append(f"grade_context: use_rag={use_rag}")
    else:
        trace.append("retrieve_rag: skipped because query is not internal")

    weather_result: WeatherResult | None = None
    if use_weather and query_scope == "weather":
        location = extract_weather_location(question)
        if location:
            try:
                weather_result = fetch_current_weather(location)
                trace.append(f"weather_tool: retrieved current weather for {weather_result.location}")
            except Exception as exc:
                trace.append(f"weather_tool: unavailable or failed ({exc})")
        else:
            trace.append("weather_tool: weather question detected but no location was found")

    web_results: list[WebSearchResult] = []
    if use_web and not weather_result and query_scope == "external":
        try:
            web_results = search_duckduckgo(question, max_results=4)
            trace.append(f"web_search: retrieved {len(web_results)} web snippets")
        except Exception:
            web_results = []
            trace.append("web_search: unavailable or failed")

    if use_rag or web_results or weather_result:
        mode_parts = []
        if use_rag:
            mode_parts.append("rag")
        if weather_result:
            mode_parts.append("weather")
        if web_results:
            mode_parts.append("web")
        mode = "+".join(mode_parts)
        prompt = build_hybrid_prompt(
            question,
            contexts if use_rag else [],
            web_results,
            weather_result=weather_result,
            mode=mode,
        )
    else:
        prompt = build_general_prompt(question)
        mode = "external_llm" if query_scope == "external" else "general_llm"
    trace.append(f"compose_prompt: mode={mode}")

    answer = generator.generate(prompt, system_prompt=system_prompt)
    trace.append("generate_answer: completed with local Ollama/HF model")
    return RagAnswer(
        question=question,
        answer=answer,
        sources=contexts if use_rag else [],
        prompt=prompt,
        model_name=llm_model_name,
        embedding_model=embedding_model_name,
        mode=mode,
        web_results=web_results,
        weather_result=weather_result,
        trace=trace,
    )


def answer_question(
    question: str,
    project_root: str | Path = PROJECT_ROOT,
    collection_name: str = DEFAULT_CHROMA_COLLECTION,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    llm_model_name: str = DEFAULT_LLM_MODEL,
    top_k: int = 5,
    llm: LocalLLM | None = None,
    use_web: bool = True,
    use_weather: bool = True,
) -> RagAnswer:
    """Answer a question through the LangGraph agent workflow.

    A manual fallback is kept so the app still gives a useful setup message if a
    user has not installed the updated requirements yet.
    """

    try:
        from .agent_graph import run_agent_graph

        return run_agent_graph(
            question=question,
            project_root=project_root,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            llm_model_name=llm_model_name,
            top_k=top_k,
            llm=llm,
            use_web=use_web,
            use_weather=use_weather,
        )
    except RuntimeError as exc:
        if "LangGraph is not installed" not in str(exc):
            raise
        fallback = _answer_question_manual(
            question=question,
            project_root=project_root,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            llm_model_name=llm_model_name,
            top_k=top_k,
            llm=llm,
            use_web=use_web,
            use_weather=use_weather,
        )
        fallback.trace.insert(0, "langgraph: not installed; used manual fallback")
        return fallback


def build_index_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Healthcare RAG Chroma index.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--collection", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--no-reset", action="store_true", help="Append to the existing collection instead of rebuilding.")
    parser.add_argument("--ocr", action="store_true", help="Use OCR for scanned PDFs and image files in knowledge_base.")
    args = parser.parse_args(argv)
    try:
        result = build_chroma_index(
            project_root=args.project_root,
            collection_name=args.collection,
            embedding_model_name=args.embedding_model,
            reset=not args.no_reset,
            use_ocr=args.ocr,
        )
    except RuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")
    print("Chroma index ready.")
    for key, value in result.items():
        print(f"{key}: {value}")


def ask_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ask the local RAG assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--collection", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--web", action="store_true", help="Use web search for external questions. This is enabled by default.")
    parser.add_argument("--no-web", action="store_true", help="Disable web search for external questions.")
    parser.add_argument("--no-weather", action="store_true", help="Disable the live no-key weather lookup tool.")
    args = parser.parse_args(argv)
    try:
        result = answer_question(
            question=args.question,
            project_root=args.project_root,
            collection_name=args.collection,
            embedding_model_name=args.embedding_model,
            llm_model_name=args.llm_model,
            top_k=args.top_k,
            use_web=not args.no_web,
            use_weather=not args.no_weather,
        )
    except RuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(result.answer)
    print(f"\nMode: {result.mode}")
    if result.sources:
        print("\nEvidence snippets:")
        for idx, source in enumerate(result.sources, start=1):
            print(f"{idx}. {source.section}: {evidence_snippet(source, 220)}")
    if result.web_results:
        print("\nWeb snippets:")
        for idx, item in enumerate(result.web_results, start=1):
            print(f"{idx}. {item.title}: {item.snippet}")
    if result.weather_result:
        print(f"\nWeather source: {result.weather_result.source_url}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or query the Healthcare RAG assistant.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the Chroma vector index.")
    build_parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    build_parser.add_argument("--collection", default=DEFAULT_CHROMA_COLLECTION)
    build_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    build_parser.add_argument("--no-reset", action="store_true")
    build_parser.add_argument("--ocr", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Ask a question using local RAG.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    ask_parser.add_argument("--collection", default=DEFAULT_CHROMA_COLLECTION)
    ask_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ask_parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--web", action="store_true")
    ask_parser.add_argument("--no-web", action="store_true")
    ask_parser.add_argument("--no-weather", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build":
        build_index_cli(
            [
                "--project-root",
                args.project_root,
                "--collection",
                args.collection,
                "--embedding-model",
                args.embedding_model,
                *(["--no-reset"] if args.no_reset else []),
                *(["--ocr"] if args.ocr else []),
            ]
        )
    elif args.command == "ask":
        ask_cli(
            [
                args.question,
                "--project-root",
                args.project_root,
                "--collection",
                args.collection,
                "--embedding-model",
                args.embedding_model,
                "--llm-model",
                args.llm_model,
                "--top-k",
                str(args.top_k),
                *(["--web"] if args.web else []),
                *(["--no-web"] if args.no_web else []),
                *(["--no-weather"] if args.no_weather else []),
            ]
        )


if __name__ == "__main__":
    main()
