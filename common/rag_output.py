"""Shared output helpers for RAG prediction artifacts."""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable


ScoreGetter = Callable[[Any], dict[str, Any]]
TextGetter = Callable[[Any], str]
MetadataGetter = Callable[[Any], dict[str, Any]]


BASE_PREDICTION_COLUMNS = [
    "question_id",
    "question",
    "gold_answer",
    "predicted_answer",
    "retrieved_sources",
    "retrieved_context_ids",
    "retrieved_context",
    "retrieval_method",
    "model_config",
]


def json_dumps(value: Any) -> str:
    """Serialize prediction fields with stable key order."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def retrieved_context_ids(items: Iterable[Any], get_metadata: MetadataGetter) -> list[str]:
    """Return chunk ids for retrieved items in final context order."""
    ids: list[str] = []
    for item in items:
        chunk_id = get_metadata(item).get("chunk_id")
        if chunk_id is not None:
            ids.append(str(chunk_id))
    return ids


def retrieved_context_records(
    items: Iterable[Any],
    *,
    get_text: TextGetter,
    get_metadata: MetadataGetter,
) -> list[dict[str, Any]]:
    """Return retrieved text plus audit metadata for prediction outputs."""
    records: list[dict[str, Any]] = []
    for item in items:
        metadata = get_metadata(item)
        records.append(
            {
                "chunk_id": str(metadata.get("chunk_id", "")),
                "source_path": metadata.get("source_path"),
                "year": metadata.get("year"),
                "month": metadata.get("month"),
                "heading": metadata.get("heading"),
                "content_type": metadata.get("content_type", "text"),
                "text": get_text(item),
            }
        )
    return records


def retrieved_source_records(
    items: Iterable[Any],
    *,
    get_metadata: MetadataGetter,
    get_scores: ScoreGetter,
) -> list[dict[str, Any]]:
    """Return compact source metadata and retrieval scores."""
    records: list[dict[str, Any]] = []
    for item in items:
        metadata = get_metadata(item)
        record = {
            "chunk_id": metadata.get("chunk_id"),
            "source_path": metadata.get("source_path"),
            "year": metadata.get("year"),
            "month": metadata.get("month"),
            "heading": metadata.get("heading"),
            "content_type": metadata.get("content_type", "text"),
        }
        record.update(get_scores(item))
        records.append(record)
    return records
