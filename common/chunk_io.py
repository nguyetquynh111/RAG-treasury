"""Shared JSONL IO for chunk records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from common.chunks import TextChunk


def save_chunks(chunks: list[TextChunk], output_path: str | Path) -> None:
    """Save chunk text and metadata as JSONL."""
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({"text": chunk.text, "metadata": chunk.metadata}) + "\n")


def load_chunks(
    chunks_path: str | Path,
    *,
    required_metadata_keys: Iterable[str] = (),
    not_found_hint: str = "Run indexing first.",
) -> list[dict[str, Any]]:
    """Load chunk records from JSONL and validate required fields."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk metadata file not found: {path}. {not_found_hint}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
            validate_chunk_record(record, path=path, line_number=line_number, keys=required_metadata_keys)
            records.append(record)

    if not records:
        raise ValueError(f"No chunk records found in {path}")
    return records


def validate_chunk_record(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    keys: Iterable[str],
) -> None:
    """Validate one JSONL chunk record."""
    if "text" not in record or "metadata" not in record:
        raise ValueError(f"Chunk record missing text/metadata in {path} line {line_number}")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Chunk metadata must be a mapping in {path} line {line_number}")
    for key in keys:
        if key not in metadata:
            raise ValueError(f"Chunk metadata missing {key!r} in {path} line {line_number}")
