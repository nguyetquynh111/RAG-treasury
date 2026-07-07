"""Shared FAISS vector-index helpers with a deterministic local fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when faiss is installed in the target env
    import faiss as _faiss
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight CI envs
    _faiss = None


FALLBACK_INDEX_FORMAT = "numpy_flat_ip_v1"


class NumpyFlatIPIndex:
    """Small in-memory inner-product index used when faiss-cpu is unavailable."""

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.vectors = np.empty((0, self.dimension), dtype="float32")

    @property
    def ntotal(self) -> int:
        """Return number of indexed vectors, matching the FAISS API used here."""
        return int(self.vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors to the index."""
        matrix = ensure_float32_matrix(vectors)
        if matrix.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors with dimension {self.dimension}, got {matrix.shape[1]}")
        self.vectors = np.vstack([self.vectors, matrix])

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k inner-product scores and vector ids."""
        query_matrix = ensure_float32_matrix(queries)
        if query_matrix.shape[1] != self.dimension:
            raise ValueError(f"Expected query dimension {self.dimension}, got {query_matrix.shape[1]}")
        if self.ntotal == 0 or top_k <= 0:
            return empty_search_result(query_matrix.shape[0], max(top_k, 0))

        k = min(int(top_k), self.ntotal)
        scores = query_matrix @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        sorted_scores = np.take_along_axis(scores, order, axis=1).astype("float32")
        return sorted_scores, order.astype("int64")


def build_inner_product_index(vectors: np.ndarray) -> Any:
    """Build a FAISS IndexFlatIP, or a compatible numpy fallback when FAISS is absent."""
    matrix = ensure_float32_matrix(vectors)
    if _faiss is not None:
        index = _faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    index = NumpyFlatIPIndex(matrix.shape[1])
    index.add(matrix)
    return index


def save_vector_index(index: Any, vectors: np.ndarray, index_path: str | Path) -> str:
    """Save the index and return the actual index backend used."""
    path = Path(index_path)
    if _faiss is not None and not isinstance(index, NumpyFlatIPIndex):
        _faiss.write_index(index, str(path))
        return "faiss"

    matrix = ensure_float32_matrix(vectors)
    with path.open("wb") as handle:
        np.savez_compressed(handle, format=FALLBACK_INDEX_FORMAT, vectors=matrix)
    return "numpy_flat_ip_fallback"


def load_vector_index(index_path: str | Path) -> Any:
    """Load a FAISS index, or the numpy fallback index format."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Vector index not found: {path}. Run indexing first.")

    if _faiss is not None:
        try:
            return _faiss.read_index(str(path))
        except Exception:
            # The file may have been created by the numpy fallback in a lightweight env.
            pass

    with path.open("rb") as handle:
        payload = np.load(handle, allow_pickle=False)
        index_format = str(payload["format"])
        if index_format != FALLBACK_INDEX_FORMAT:
            raise ValueError(f"Unsupported fallback index format: {index_format}")
        vectors = payload["vectors"].astype("float32")

    index = NumpyFlatIPIndex(vectors.shape[1])
    index.add(vectors)
    return index


def ensure_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    """Validate and coerce a matrix to float32."""
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D vector matrix, got shape {matrix.shape}")
    return matrix


def empty_search_result(number_queries: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an empty FAISS-like search result."""
    return (
        np.empty((number_queries, top_k), dtype="float32"),
        -np.ones((number_queries, top_k), dtype="int64"),
    )
