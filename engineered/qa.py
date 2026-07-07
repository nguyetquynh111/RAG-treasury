"""Engineered retrieval QA runner for OfficeQA Treasury questions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from engineered.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from engineered.dataset import load_filtered_officeqa
from engineered.generation import RAGAnswerGenerator
from engineered.query import QueryFilters, parse_query_filters
from engineered.retrieval import Candidate, HybridRetriever, RERANKING_METHOD, tokenize
from common.rag_output import (
    json_dumps,
    retrieved_context_ids,
    retrieved_context_records,
    retrieved_source_records,
)


RETRIEVAL_METHOD = "hybrid_vector_bm25_rerank_rag_generate"
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def run_qa(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Generate engineered predictions without evaluation."""
    config = load_config(config_path)
    csv_path = resolve_path(config_path, config["csv_path"])
    output_dir = resolve_path(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_filtered_officeqa(csv_path, config["selected_years"])
    retriever = HybridRetriever(config_path)
    generator = RAGAnswerGenerator(config, extractive_fallback=grounded_answer)
    model_config = summarize_model_config(config, RERANKING_METHOD)

    predictions_path = output_dir / "predictions.csv"
    logs_path = output_dir / "retrieval_logs.jsonl"
    columns = prediction_columns()
    total_questions = len(questions)
    with predictions_path.open("w", encoding="utf-8", newline="") as predictions_handle, logs_path.open(
        "w", encoding="utf-8"
    ) as logs_handle:
        writer = csv.DictWriter(predictions_handle, fieldnames=columns)
        writer.writeheader()
        flush_output(predictions_handle)
        for completed_count, row in enumerate(questions, start=1):
            filters = query_filters_with_row_fallback(
                row.question,
                config["selected_years"],
                row.row_year,
                row.row_month,
            )
            retrieved, diagnostics = retriever.retrieve(row.question, filters)
            predicted_answer = generator.generate(row.question, retrieved)
            final_context_ids = retrieved_context_ids(retrieved, lambda candidate: candidate.metadata)
            final_sources = retrieved_source_records(
                retrieved,
                get_metadata=lambda candidate: candidate.metadata,
                get_scores=engineered_scores,
            )
            final_context = retrieved_context_records(
                retrieved,
                get_text=lambda candidate: candidate.text,
                get_metadata=lambda candidate: candidate.metadata,
            )

            prediction_row = {
                "question_id": row.question_id,
                "question": row.question,
                "gold_answer": row.gold_answer,
                "predicted_answer": predicted_answer,
                "selected_years": json_dumps(config["selected_years"]),
                "detected_year": filters.year,
                "detected_month": filters.month,
                "retrieved_sources": json_dumps(final_sources),
                "retrieved_context_ids": json_dumps(final_context_ids),
                "retrieved_context": json_dumps(final_context),
                "retrieval_method": RETRIEVAL_METHOD,
                "model_config": model_config,
            }
            writer.writerow(prediction_row)
            flush_output(predictions_handle)

            log_row = {
                "question_id": row.question_id,
                "detected_year": filters.year,
                "detected_month": filters.month,
                "number_vector_candidates": diagnostics["number_vector_candidates"],
                "number_bm25_candidates": diagnostics["number_bm25_candidates"],
                "number_merged_candidates": diagnostics["number_merged_candidates"],
                "fusion_method": diagnostics["fusion_method"],
                "reranker_backend": diagnostics["reranker_backend"],
                "answer_backend": generator.actual_backend,
                "answer_model": generator.actual_model,
                "final_context_ids": final_context_ids,
                "final_sources": final_sources,
            }
            logs_handle.write(json.dumps(log_row) + "\n")
            flush_output(logs_handle)
            print(
                f"[engineered] wrote {completed_count}/{total_questions} question_id={row.question_id} "
                f"to {predictions_path}",
                flush=True,
            )

    return predictions_path


def flush_output(handle: Any) -> None:
    """Flush incremental outputs so completed answers survive interrupted runs."""
    handle.flush()
    os.fsync(handle.fileno())


def grounded_answer(question: str, retrieved_chunks: list[Candidate], max_sentences: int = 2) -> str:
    """Return a concise extractive answer from retrieved context, or NOT_FOUND."""
    if not retrieved_chunks:
        return "NOT_FOUND"

    question_terms = set(tokenize(question, drop_stopwords=True))
    if not question_terms:
        return "NOT_FOUND"

    candidates: list[tuple[int, float, str]] = []
    for rank, chunk in enumerate(retrieved_chunks):
        for sentence in split_evidence_units(chunk.text):
            sentence_terms = set(tokenize(sentence, drop_stopwords=True))
            overlap = len(question_terms & sentence_terms)
            if overlap == 0:
                continue
            score = overlap + max(0.0, chunk.rerank_score) - (rank * 0.01)
            candidates.append((overlap, score, sentence))

    if not candidates:
        return "NOT_FOUND"

    required_overlap = max(1, min(3, math.ceil(len(question_terms) * 0.2)))
    best_overlap = max(overlap for overlap, _, _ in candidates)
    if best_overlap < required_overlap:
        return "NOT_FOUND"

    candidates.sort(key=lambda item: (-item[1], -item[0], len(item[2])))
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, sentence in candidates:
        normalized = " ".join(sentence.split())
        if normalized in seen:
            continue
        selected.append(normalized)
        seen.add(normalized)
        if len(selected) == max_sentences:
            break

    return " ".join(selected) if selected else "NOT_FOUND"


def split_evidence_units(text: str) -> list[str]:
    """Split retrieved context into concise sentence or table-row evidence units."""
    units: list[str] = []
    for raw in SENTENCE_PATTERN.split(text):
        clean = " ".join(raw.split())
        if len(clean) < 20:
            continue
        if len(clean.split()) > 80:
            clean = " ".join(clean.split()[:80])
        units.append(clean)
    return units



def query_filters_with_row_fallback(
    question: str,
    selected_years: list[int],
    row_year: int,
    row_month: int | None,
) -> QueryFilters:
    """Detect year/month from the query, then fall back to answer-key metadata."""
    detected = parse_query_filters(question, selected_years)
    return QueryFilters(
        year=detected.year if detected.year is not None else row_year,
        month=detected.month if detected.month is not None else row_month,
    )


def engineered_scores(candidate: Candidate) -> dict[str, Any]:
    """Return all retrieval score fields for one engineered candidate."""
    return {
        "vector_score": candidate.vector_score,
        "bm25_score": candidate.bm25_score,
        "vector_rank": candidate.vector_rank,
        "bm25_rank": candidate.bm25_rank,
        "fused_score": candidate.fused_score,
        "keyword_score": candidate.keyword_score,
        "cross_encoder_score": candidate.cross_encoder_score,
        "rerank_score": candidate.rerank_score,
    }

def summarize_model_config(config: dict[str, Any], reranking_method: str) -> str:
    """Summarize retrieval settings for prediction output."""
    embedding = config.get("embedding", {})
    summary = {
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "bm25_top_k": config["bm25_top_k"],
        "vector_top_k": config["vector_top_k"],
        "final_top_k": config["final_top_k"],
        "fusion_method": config.get("fusion_method", "rrf"),
        "embedding_model": embedding.get("model_name", embedding.get("backend", "unknown")),
        "reranking_method": reranking_method,
        "reranker": config.get("reranker", {}),
        "generation": config.get("generation", {"backend": "deepinfra"}),
    }
    return json.dumps(summary, sort_keys=True)


def prediction_columns() -> list[str]:
    """Return the required engineered prediction schema."""
    return [
        "question_id",
        "question",
        "gold_answer",
        "predicted_answer",
        "selected_years",
        "detected_year",
        "detected_month",
        "retrieved_sources",
        "retrieved_context_ids",
        "retrieved_context",
        "retrieval_method",
        "model_config",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run engineered Treasury retrieval QA predictions only.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to engineered YAML config.")
    args = parser.parse_args()

    predictions_path = run_qa(args.config)
    print(f"Wrote engineered predictions to {predictions_path}")


if __name__ == "__main__":
    main()
