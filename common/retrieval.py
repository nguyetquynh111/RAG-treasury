"""Shared vector retrieval with optional metadata filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.chunking import load_chunks
from common.config import DEFAULT_CONFIG_PATH, load_pipeline_config, resolve_path
from common.embeddings import Embedder
from common.index import build_index
from common.vector_store import load_vector_index


INDEX_FILES = ("index.faiss", "chunks.jsonl", "embeddings.npy", "manifest.json")
REQUIRED_CHUNK_METADATA = ("year", "month", "source_path", "chunk_id")


class VectorRetriever:
    """Retriever over the single shared index.

    Baseline calls ``retrieve(question)``. Engineered mode passes year/month or
    exact source date pairs to restrict the same index at retrieval time.
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        mode: str = "baseline",
        rebuild: bool = False,
        auto_build: bool = True,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_pipeline_config(self.config_path, mode=mode)
        self.index_dir = resolve_path(self.config_path, self.config["index_dir"])

        if rebuild or (auto_build and not self._artifacts_exist()):
            build_index(self.config_path)

        self.index = load_vector_index(self.index_dir / "index.faiss")
        self.chunks = load_chunks(
            self.index_dir / "chunks.jsonl",
            required_metadata_keys=REQUIRED_CHUNK_METADATA,
            not_found_hint="Build the shared index first with: python -m common.index --config config/config.yaml",
        )
        self.embeddings = np.load(self.index_dir / "embeddings.npy")
        self.embedder = Embedder(self.config)
        self._validate_artifacts()

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        year: int | None = None,
        month: int | None = None,
        date_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k chunks from the shared index, optionally filtered by metadata."""
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        k = int(top_k or self.config["top_k"])
        if k <= 0:
            raise ValueError("top_k must be positive.")

        allowed_indices = matching_indices(self.chunks, year=year, month=month, date_pairs=date_pairs)
        if not allowed_indices:
            return []

        query = self.embedder.encode([question])[0]
        ranked = ranked_vector_search(
            index=self.index,
            embeddings=self.embeddings,
            query=query,
            allowed_indices=allowed_indices,
            top_k=k,
        )
        return [self._result(chunk_index, score) for chunk_index, score in ranked]

    def _result(self, chunk_index: int, score: float) -> dict[str, Any]:
        chunk = self.chunks[chunk_index]
        return {
            "score": score,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
        }

    def _artifacts_exist(self) -> bool:
        return all((self.index_dir / filename).exists() for filename in INDEX_FILES)

    def _validate_artifacts(self) -> None:
        if len(self.chunks) != self.index.ntotal:
            raise ValueError(
                f"Index/chunk count mismatch: index has {self.index.ntotal}, chunks file has {len(self.chunks)}"
            )
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                f"Embedding/chunk count mismatch: embeddings has {len(self.embeddings)}, chunks has {len(self.chunks)}"
            )


def matching_indices(
    chunks: list[dict[str, Any]],
    *,
    year: int | None = None,
    month: int | None = None,
    date_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
) -> list[int]:
    """Return chunk indices matching optional year/month metadata filters."""
    pair_filter = {(int(pair_year), int(pair_month)) for pair_year, pair_month in date_pairs or []}
    matches: list[int] = []

    for index, chunk in enumerate(chunks):
        metadata = chunk["metadata"]
        chunk_pair = (int(metadata["year"]), int(metadata["month"]))
        if pair_filter and chunk_pair not in pair_filter:
            continue
        if not pair_filter and year is not None and chunk_pair[0] != int(year):
            continue
        if not pair_filter and month is not None and chunk_pair[1] != int(month):
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
    """Return ranked vector matches, using FAISS directly when no filter is active."""
    if top_k <= 0 or not allowed_indices:
        return []

    k = min(int(top_k), len(allowed_indices))
    query_vector = np.asarray(query, dtype="float32").reshape(1, -1)

    if allowed_indices == list(range(len(embeddings))):
        scores, ids = index.search(query_vector, min(k, index.ntotal))
        return [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx >= 0]

    allowed_array = np.asarray(allowed_indices, dtype="int64")
    scores = embeddings[allowed_array] @ query_vector.reshape(-1)
    order = np.argsort(-scores)[:k]
    return [(int(allowed_array[position]), float(scores[position])) for position in order]
