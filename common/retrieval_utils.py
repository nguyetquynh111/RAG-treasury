"""Shared retrieval helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


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


def ranked_vector_search(
    *,
    index: Any,
    embeddings: np.ndarray,
    query: np.ndarray,
    allowed_indices: list[int],
    top_k: int,
) -> list[tuple[int, float]]:
    """Return ranked vector matches, using the full index when filters allow it."""
    if top_k <= 0 or not allowed_indices:
        return []

    k = min(int(top_k), len(allowed_indices))
    query_vector = np.asarray(query, dtype="float32").reshape(1, -1)

    all_indices = list(range(len(embeddings)))
    if allowed_indices == all_indices:
        scores, ids = index.search(query_vector, min(k, index.ntotal))
        return [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx >= 0]

    allowed_array = np.asarray(allowed_indices, dtype="int64")
    scores = embeddings[allowed_array] @ query_vector.reshape(-1)
    order = np.argsort(-scores)[:k]
    return [(int(allowed_array[position]), float(scores[position])) for position in order]
