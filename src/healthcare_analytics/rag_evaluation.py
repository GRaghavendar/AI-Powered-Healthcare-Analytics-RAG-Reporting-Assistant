"""Small RAG evaluation runner for assistant validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, PROJECT_ROOT
from .rag import answer_question

DEFAULT_EVALUATION_CASES = [
    {
        "question": "What are the main trauma metrics in this project?",
        "expected_keywords": ["fatality", "icu", "ventilator", "length of stay", "claims"],
    },
    {
        "question": "How does the project protect PHI?",
        "expected_keywords": ["synthetic", "aggregate", "phi", "patient", "privacy"],
    },
    {
        "question": "How is the Chroma RAG workflow structured?",
        "expected_keywords": ["embedding", "chroma", "retrieval", "chunks", "ollama"],
    },
    {
        "question": "What validation checks happen before reporting?",
        "expected_keywords": ["missing", "duplicate", "orphan", "schema", "facility"],
    },
    {
        "question": "Give a summary of the audit report for each facility.",
        "expected_keywords": ["facility", "hospital", "adult", "pediatric", "total"],
    },
]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _keyword_score(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return round(matches / len(expected_keywords), 3)


def _groundedness_score(answer: str, evidence_text: str) -> float:
    answer_tokens = _tokens(answer)
    evidence_tokens = _tokens(evidence_text)
    if not answer_tokens or not evidence_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return round(len(overlap) / len(answer_tokens), 3)


def _retrieval_top_k_score(source_scores: list[float]) -> float:
    if not source_scores:
        return 0.0
    best_distance = min(source_scores)
    return round(max(0.0, min(1.0, 1.0 - (best_distance / 2.0))), 3)


def run_evaluation(
    project_root: str | Path = PROJECT_ROOT,
    questions: list[str] | None = None,
    output_path: str | Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> list[dict]:
    rows = []
    if questions:
        evaluation_cases = [{"question": question, "expected_keywords": []} for question in questions]
    else:
        evaluation_cases = DEFAULT_EVALUATION_CASES

    for case in evaluation_cases:
        question = case["question"]
        expected_keywords = case.get("expected_keywords", [])
        result = answer_question(
            question=question,
            project_root=project_root,
            embedding_model_name=embedding_model,
            llm_model_name=llm_model,
            top_k=5,
        )
        evidence_text = "\n\n".join(source.text for source in result.sources)
        source_scores = [source.score for source in result.sources]
        keyword_score = _keyword_score(result.answer, expected_keywords)
        groundedness = _groundedness_score(result.answer, evidence_text)
        retrieval_score = _retrieval_top_k_score(source_scores)
        sources_retrieved = bool(result.sources)
        passed = sources_retrieved and keyword_score >= 0.4 and groundedness >= 0.08
        rows.append(
            {
                "question": question,
                "expected_keywords": expected_keywords,
                "answer": result.answer,
                "mode": result.mode,
                "sources": [
                    {
                        "source": source.source,
                        "section": source.section,
                        "score": source.score,
                        "source_type": source.source_type,
                    }
                    for source in result.sources
                ],
                "sources_retrieved": sources_retrieved,
                "source_count": len(result.sources),
                "keyword_coverage_score": keyword_score,
                "answer_groundedness_score": groundedness,
                "retrieval_top_k_score": retrieval_score,
                "pass": passed,
                "llm_model": result.model_name,
                "embedding_model": result.embedding_model,
            }
        )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run sample RAG evaluation questions.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "rag_evaluation.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    args = parser.parse_args(argv)
    try:
        rows = run_evaluation(
            project_root=args.project_root,
            output_path=args.output,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
        )
    except RuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Wrote {len(rows)} evaluation rows to {args.output}")
