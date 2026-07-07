"""Baseline top-k retrieval with optional year/month filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from baseline.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from common.embeddings import Embedder
from common.chunk_io import load_chunks
from common.retrieval_utils import matching_indices
from common.vector_index import load_vector_index


class Retriever:
    """FAISS retriever with metadata filtering support."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.output_dir = resolve_path(self.config_path, self.config["output_dir"])

        index_path = self.output_dir / "index.faiss"
        chunks_path = self.output_dir / "chunks.jsonl"
        embeddings_path = self.output_dir / "embeddings.npy"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}. Run indexing first.")
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embedding matrix not found: {embeddings_path}. Run indexing first.")

        self.index = load_vector_index(index_path)
        self.chunks = load_chunks(chunks_path, not_found_hint="Run indexing first.")
        self.embeddings = np.load(embeddings_path)
        if len(self.chunks) != self.index.ntotal:
            raise ValueError(
                f"Index/chunk count mismatch: index has {self.index.ntotal}, chunks file has {len(self.chunks)}"
            )
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                f"Embedding/chunk count mismatch: embeddings has {len(self.embeddings)}, chunks has {len(self.chunks)}"
            )

        self.embedder = Embedder(self.config)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        year: int | None = None,
        month: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks, optionally restricted to year/month metadata."""
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        k = int(top_k or self.config["top_k"])
        if k <= 0:
            raise ValueError("top_k must be positive.")

        query = self.embedder.encode([question])[0]
        allowed_indices = self._matching_indices(year=year, month=month)
        if not allowed_indices:
            return []

        if len(allowed_indices) == len(self.chunks):
            scores, ids = self.index.search(query.reshape(1, -1), min(k, self.index.ntotal))
            pairs = [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx >= 0]
        else:
            matrix = self.embeddings[np.array(allowed_indices)]
            scores = matrix @ query
            order = np.argsort(-scores)[:k]
            pairs = [(allowed_indices[int(position)], float(scores[int(position)])) for position in order]

        results = []
        for chunk_index, score in pairs[:k]:
            record = self.chunks[chunk_index]
            results.append(
                {
                    "score": score,
                    "text": record["text"],
                    "metadata": record["metadata"],
                }
            )
        return results

    def _matching_indices(self, year: int | None = None, month: int | None = None) -> list[int]:
        return matching_indices(self.chunks, year=year, month=month)
