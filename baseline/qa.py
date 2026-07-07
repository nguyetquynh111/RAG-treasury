"""Baseline RAG runner for OfficeQA Treasury questions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from baseline.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from baseline.dataset import load_filtered_officeqa
from baseline.generation import BaselineRAGAnswerGenerator
from baseline.retrieval import Retriever
from common.rag_output import (
    retrieved_context_ids,
    retrieved_context_records,
    retrieved_source_records,
)


RETRIEVAL_METHOD = "faiss_vector_top_k_rag_generate"


def run_qa(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Run baseline RAG generation over selected OfficeQA rows and save predictions."""
    config = load_config(config_path)
    csv_path = resolve_path(config_path, config["csv_path"])
    output_dir = resolve_path(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_key = load_filtered_officeqa(csv_path, config["selected_years"])
    retriever = Retriever(config_path)
    generator = BaselineRAGAnswerGenerator(config, extractive_fallback=extractive_answer)
    model_config = summarize_model_config(config, generator.actual_backend, generator.actual_model)

    predictions_path = output_dir / "predictions.csv"
    logs_path = output_dir / "retrieval_logs.jsonl"
    columns = prediction_columns()
    total_questions = len(answer_key)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle, logs_path.open(
        "w", encoding="utf-8"
    ) as logs_handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        flush_csv(handle)
        for completed_count, (row_id, row) in enumerate(answer_key.iterrows(), start=1):
            question = str(row["baseline_question"])
            question_id = str(row.get("question_id", row_id))
            year = int(row["baseline_year"])
            month = None if pd.isna(row["baseline_month"]) else int(row["baseline_month"])
            retrieved = retriever.retrieve(question, top_k=config["top_k"], year=year, month=month)
            predicted_answer = generator.generate(question, retrieved)
            final_context_ids = retrieved_context_ids(retrieved, lambda chunk: chunk.get("metadata", {}))
            final_sources = retrieved_source_records(
                retrieved,
                get_metadata=lambda chunk: chunk.get("metadata", {}),
                get_scores=baseline_scores,
            )
            final_context = retrieved_context_records(
                retrieved,
                get_text=lambda chunk: str(chunk.get("text", "")),
                get_metadata=lambda chunk: chunk.get("metadata", {}),
            )

            prediction_row = {
                "question_id": question_id,
                "question": question,
                "gold_answer": row["baseline_answer"],
                "predicted_answer": predicted_answer,
                "detected_year": year,
                "detected_month": month,
                "retrieval_method": RETRIEVAL_METHOD,
            }
            writer.writerow(prediction_row)
            flush_csv(handle)

            log_row = {
                "question_id": question_id,
                "selected_years": config["selected_years"],
                "detected_year": year,
                "detected_month": month,
                "retrieval_method": RETRIEVAL_METHOD,
                "model_config": json.loads(model_config),
                "final_context_ids": final_context_ids,
                "final_sources": final_sources,
                "final_context": final_context,
            }
            logs_handle.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            flush_csv(logs_handle)
            print(
                f"[baseline] wrote {completed_count}/{total_questions} question_id={question_id} "
                f"to {predictions_path}",
                flush=True,
            )
    return predictions_path


def flush_csv(handle: Any) -> None:
    """Flush incremental CSV output so completed answers survive interrupted runs."""
    handle.flush()
    os.fsync(handle.fileno())


def extractive_answer(question: str, retrieved_chunks: list[dict[str, Any]], max_sentences: int = 2) -> str:
    """Return a cited fallback answer using sentences from retrieved chunks only."""
    if not retrieved_chunks:
        return "NOT_FOUND"

    question_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", question)
        if len(term) > 2
    }
    candidates: list[tuple[int, int, str]] = []
    for source_index, chunk in enumerate(retrieved_chunks, start=1):
        sentences = re.split(r"(?<=[.!?])\s+|\n+", str(chunk.get("text", "")))
        for sentence in sentences:
            clean = " ".join(sentence.split())
            if not clean:
                continue
            sentence_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", clean)}
            overlap = len(question_terms & sentence_terms)
            candidates.append((overlap, source_index, clean))

    if not candidates:
        snippet = " ".join(str(retrieved_chunks[0].get("text", "")).split()[:80])
        return f"{snippet} [S1]" if snippet else "NOT_FOUND"

    candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    selected = [f"{sentence} [S{source_index}]" for _, source_index, sentence in candidates[:max_sentences]]
    return " ".join(selected) if selected else "NOT_FOUND"


def baseline_scores(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return score fields for one baseline retrieved chunk."""
    return {"vector_score": chunk.get("score")}


def summarize_model_config(config: dict[str, Any], backend: str, model: str) -> str:
    """Summarize baseline RAG settings for prediction artifacts."""
    embedding = config.get("embedding", {})
    return json.dumps(
        {
            "pipeline": "baseline_rag",
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "embedding_model": embedding.get("model_name", embedding.get("backend", "unknown")),
            "retrieval": "faiss_vector_top_k",
            "top_k": config["top_k"],
            "generation_backend": backend,
            "generation_model": model,
            "generation": config.get("generation", {"backend": "deepinfra"}),
        },
        sort_keys=True,
    )


def prediction_columns() -> list[str]:
    """Return the baseline prediction schema."""
    return [
        "question_id",
        "question",
        "gold_answer",
        "predicted_answer",
        "detected_year",
        "detected_month",
        "retrieval_method",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline Treasury RAG QA.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to baseline YAML config.")
    args = parser.parse_args()

    predictions_path = run_qa(args.config)
    print(f"Wrote baseline RAG predictions to {predictions_path}")


if __name__ == "__main__":
    main()
