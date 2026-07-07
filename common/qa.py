"""Run Treasury RAG QA for baseline and engineered modes."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from common.config import DEFAULT_CONFIG_PATH, PIPELINE_MODES, load_pipeline_config, resolve_path
from common.dataset import OfficeQARow, load_filtered_officeqa_rows
from common.generation import (
    BASE_CITATION_FIELDS,
    METADATA_CITATION_FIELDS,
    VectorRAGAnswerGenerator,
    extractive_answer,
)
from common.mode import ModeBehavior, RetrievalPlan
from common.query import QueryFilters, parse_query_filters
from common.rag_output import retrieved_context_ids, retrieved_context_records, retrieved_source_records
from common.retrieval import VectorRetriever


class RAGAnswerGenerator(VectorRAGAnswerGenerator):
    """Answer generator configured by mode settings."""

    def __init__(self, config: dict[str, Any]) -> None:
        citation_fields = METADATA_CITATION_FIELDS if config["metadata_enabled"] else BASE_CITATION_FIELDS
        super().__init__(
            config,
            extractive_fallback=extractive_answer,
            citation_fields=citation_fields,
        )


def run_qa(config_path: str | Path = DEFAULT_CONFIG_PATH, *, mode: str) -> Path:
    """Generate predictions for either baseline or engineered mode."""
    config = load_pipeline_config(config_path, mode=mode)
    mode_behavior = load_mode_behavior(mode)
    csv_path = resolve_path(config_path, config["csv_path"])
    output_dir = resolve_path(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_filtered_officeqa_rows(csv_path, config["selected_years"])
    retriever = VectorRetriever(config_path, mode=mode)
    generator = RAGAnswerGenerator(config)
    model_config = summarize_model_config(config, generator.actual_backend, generator.actual_model)

    predictions_path = output_dir / "predictions.csv"
    logs_path = output_dir / "retrieval_logs.jsonl"
    write_predictions(
        rows=rows,
        config=config,
        mode_behavior=mode_behavior,
        retriever=retriever,
        generator=generator,
        model_config=model_config,
        predictions_path=predictions_path,
        logs_path=logs_path,
    )
    return predictions_path


def load_mode_behavior(mode: str) -> ModeBehavior:
    """Load the small mode-specific retrieval module."""
    module = importlib.import_module(f"{mode}.retrieval")
    require_mode_functions(module, mode)
    return module


def require_mode_functions(module: ModuleType, mode: str) -> None:
    """Validate the retrieval behavior module at startup."""
    required_names = ("RETRIEVAL_METHOD", "build_retrieval_plan", "retrieve", "extra_log_fields")
    missing = [name for name in required_names if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"{mode}.retrieval is missing required attributes: {missing}")


def write_predictions(
    *,
    rows: list[OfficeQARow],
    config: dict[str, Any],
    mode_behavior: ModeBehavior,
    retriever: VectorRetriever,
    generator: RAGAnswerGenerator,
    model_config: str,
    predictions_path: Path,
    logs_path: Path,
) -> None:
    """Write prediction CSV and retrieval JSONL sidecar."""
    mode = config["mode"]
    with predictions_path.open("w", encoding="utf-8", newline="") as predictions_handle, logs_path.open(
        "w", encoding="utf-8"
    ) as logs_handle:
        writer = csv.DictWriter(predictions_handle, fieldnames=prediction_columns())
        writer.writeheader()
        flush_output(predictions_handle)

        for completed_count, row in enumerate(rows, start=1):
            plan = mode_behavior.build_retrieval_plan(row, config)
            retrieved = mode_behavior.retrieve(retriever, row.question, config, plan)
            predicted_answer = generator.generate(row.question, retrieved)
            writer.writerow(prediction_row(row, predicted_answer, plan, mode_behavior))
            flush_output(predictions_handle)
            logs_handle.write(json.dumps(log_row(row, retrieved, plan, config, mode_behavior, model_config), ensure_ascii=False) + "\n")
            flush_output(logs_handle)
            print(
                f"[{mode}] wrote {completed_count}/{len(rows)} question_id={row.question_id} to {predictions_path}",
                flush=True,
            )


def prediction_row(
    row: OfficeQARow,
    predicted_answer: str,
    plan: RetrievalPlan,
    mode_behavior: ModeBehavior,
) -> dict[str, Any]:
    """Return one prediction CSV row."""
    return {
        "question_id": row.question_id,
        "question": row.question,
        "gold_answer": row.gold_answer,
        "predicted_answer": predicted_answer,
        "detected_year": plan.year,
        "detected_month": plan.month,
        "retrieval_method": mode_behavior.RETRIEVAL_METHOD,
    }


def log_row(
    row: OfficeQARow,
    retrieved: list[dict[str, Any]],
    plan: RetrievalPlan,
    config: dict[str, Any],
    mode_behavior: ModeBehavior,
    model_config: str,
) -> dict[str, Any]:
    """Return one retrieval-log JSON object."""
    record = {
        "question_id": row.question_id,
        "selected_years": config["selected_years"],
        "detected_year": plan.year,
        "detected_month": plan.month,
        "retrieval_method": mode_behavior.RETRIEVAL_METHOD,
        "model_config": json.loads(model_config),
        "final_context_ids": retrieved_context_ids(retrieved, lambda chunk: chunk.get("metadata", {})),
        "final_sources": retrieved_source_records(
            retrieved,
            get_metadata=lambda chunk: chunk.get("metadata", {}),
            get_scores=vector_scores,
        ),
        "final_context": retrieved_context_records(
            retrieved,
            get_text=lambda chunk: str(chunk.get("text", "")),
            get_metadata=lambda chunk: chunk.get("metadata", {}),
        ),
    }
    record.update(mode_behavior.extra_log_fields(plan))
    return record


def query_filters_with_row_fallback(
    question: str,
    selected_years: list[int],
    row_year: int,
    row_month: int | None,
    source_date_pairs: tuple[tuple[int, int], ...] = (),
) -> QueryFilters:
    """Compatibility helper used by tests/notebooks."""
    detected = parse_query_filters(question, selected_years)
    return QueryFilters(
        year=row_year if row_year is not None else detected.year,
        month=row_month if row_month is not None else None,
        date_pairs=tuple(source_date_pairs),
    )


def flush_output(handle: Any) -> None:
    """Flush incremental artifacts so interrupted runs keep completed rows."""
    handle.flush()
    os.fsync(handle.fileno())


def vector_scores(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return score fields for one retrieved chunk."""
    return {"vector_score": chunk.get("score")}


def summarize_model_config(config: dict[str, Any], backend: str, model: str) -> str:
    """Summarize run settings for the retrieval log."""
    embedding = config.get("embedding", {})
    return json.dumps(
        {
            "pipeline": "metadata_rag" if config["metadata_enabled"] else "baseline_rag",
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "embedding_model": embedding.get("model_name", embedding.get("backend", "unknown")),
            "retrieval": "faiss_vector_top_k",
            "top_k": config["top_k"],
            "metadata_filtering": config["metadata_enabled"],
            "generation_backend": backend,
            "generation_model": model,
            "generation": config.get("generation", {"backend": "deepinfra"}),
        },
        sort_keys=True,
    )


def prediction_columns() -> list[str]:
    """Return the prediction CSV schema."""
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
    """CLI entrypoint for running one QA mode."""
    parser = argparse.ArgumentParser(description="Run Treasury RAG QA with the shared index.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to shared YAML config.")
    parser.add_argument("--mode", choices=sorted(PIPELINE_MODES), required=True, help="Run mode to execute.")
    args = parser.parse_args()

    predictions_path = run_qa(args.config, mode=args.mode)
    print(f"Wrote {args.mode} predictions to {predictions_path}")


if __name__ == "__main__":
    main()
