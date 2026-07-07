"""Shared retrieval helpers."""

from __future__ import annotations

from typing import Any


def matching_indices(
    chunks: list[dict[str, Any]],
    *,
    year: int | None = None,
    month: int | None = None,
) -> list[int]:
    """Return chunk indices matching optional year/month metadata filters."""
    matches: list[int] = []
    for index, chunk in enumerate(chunks):
        metadata = chunk["metadata"]
        if year is not None and int(metadata["year"]) != int(year):
            continue
        if month is not None and int(metadata["month"]) != int(month):
            continue
        matches.append(index)
    return matches
