"""Hybrid vector + BM25 retrieval with fusion and optional cross-encoder reranking."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.embeddings import Embedder
from engineered.chunking import chunk_documents
from engineered.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from engineered.dataset import load_treasury_documents
from common.chunk_io import load_chunks, save_chunks
from common.retrieval_utils import matching_indices
from common.vector_index import build_inner_product_index, load_vector_index, save_vector_index
from engineered.query import QueryFilters


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
RERANKING_METHOD = "rrf_fusion_cross_encoder_with_deterministic_fallback"


@dataclass
class Candidate:
    text: str
    metadata: dict[str, Any]
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    fused_score: float = 0.0
    keyword_score: float = 0.0
    cross_encoder_score: float | None = None
    rerank_score: float = 0.0


def build_index(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, str | int]:
    """Create engineered chunks, embeddings, FAISS index, and manifest."""
    config = load_config(config_path)
    data_dir = resolve_path(config_path, config["data_dir"])
    output_dir = resolve_path(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = load_treasury_documents(data_dir, config["selected_years"])
    chunks = chunk_documents(documents, config["chunk_size"], config["chunk_overlap"])

    embedder = Embedder(config)
    vectors = embedder.encode([chunk.text for chunk in chunks])
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {vectors.shape}")

    index = build_inner_product_index(vectors)

    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.jsonl"
    vectors_path = output_dir / "embeddings.npy"
    manifest_path = output_dir / "manifest.json"

    index_backend = save_vector_index(index, vectors, index_path)
    save_chunks(chunks, chunks_path)
    np.save(vectors_path, vectors)

    manifest = {
        "vector_db_type": "faiss",
        "index_backend": index_backend,
        "embedding_backend": embedder.actual_backend,
        "embedding_model_name": embedder.model_name,
        "embedding_dim": int(vectors.shape[1]),
        "normalize_embeddings": bool(config.get("embedding", {}).get("normalize", True)),
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "selected_years": config["selected_years"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "bm25_top_k": config["bm25_top_k"],
        "vector_top_k": config["vector_top_k"],
        "final_top_k": config["final_top_k"],
        "fusion_method": config.get("fusion_method", "rrf"),
        "reranking_method": RERANKING_METHOD,
        "reranker_backend": config.get("reranker", {}).get("backend", "cross_encoder"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "index_path": str(index_path),
        "chunks_path": str(chunks_path),
        "manifest_path": str(manifest_path),
    }


class HybridRetriever:
    """Hybrid retriever applying metadata filters to vector and BM25 search."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH, rebuild: bool = False):
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.output_dir = resolve_path(self.config_path, self.config["output_dir"])

        if rebuild or not self._artifacts_exist():
            build_index(self.config_path)

        self.index = load_vector_index(self.output_dir / "index.faiss")
        self.chunks = load_chunks(
            self.output_dir / "chunks.jsonl",
            required_metadata_keys=("year", "month", "source_path", "heading", "chunk_id"),
            not_found_hint="Build the engineered index first.",
        )
        for record in self.chunks:
            record["metadata"].setdefault("content_type", "text")
        self.embeddings = np.load(self.output_dir / "embeddings.npy")
        if len(self.chunks) != self.index.ntotal:
            raise ValueError(f"Index/chunk count mismatch: {self.index.ntotal} vs {len(self.chunks)}")
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(f"Embedding/chunk count mismatch: {len(self.embeddings)} vs {len(self.chunks)}")

        self.embedder = Embedder(self.config)
        self._tokenized_chunks = [tokenize(record["text"]) for record in self.chunks]
        self._bm25 = BM25Index(self._tokenized_chunks)
        self.fusion_method = str(self.config.get("fusion_method", "rrf")).lower()
        self.rrf_k = int(self.config.get("rrf_k", 60))
        self.reranker = CrossEncoderReranker(self.config)

    def retrieve(self, question: str, filters: QueryFilters) -> tuple[list[Candidate], dict[str, Any]]:
        """Return final reranked candidates and retrieval diagnostics."""
        allowed_indices = self._matching_indices(filters)
        if not allowed_indices:
            return [], {
                "number_vector_candidates": 0,
                "number_bm25_candidates": 0,
                "number_merged_candidates": 0,
                "fusion_method": self.fusion_method,
                "reranker_backend": self.reranker.actual_backend,
            }

        vector_candidates = self._vector_search(question, allowed_indices, self.config["vector_top_k"])
        bm25_candidates = self._bm25_search(question, allowed_indices, self.config["bm25_top_k"])
        merged = self._merge_candidates(vector_candidates, bm25_candidates)
        self._fuse_candidates(merged)
        final = self._rerank(question, merged, self.config["final_top_k"])

        return final, {
            "number_vector_candidates": len(vector_candidates),
            "number_bm25_candidates": len(bm25_candidates),
            "number_merged_candidates": len(merged),
            "fusion_method": self.fusion_method,
            "reranker_backend": self.reranker.actual_backend,
        }

    def _artifacts_exist(self) -> bool:
        return all(
            (self.output_dir / filename).exists()
            for filename in ["index.faiss", "chunks.jsonl", "embeddings.npy", "manifest.json"]
        )

    def _matching_indices(self, filters: QueryFilters) -> list[int]:
        return matching_indices(self.chunks, year=filters.year, month=filters.month)

    def _vector_search(self, question: str, allowed_indices: list[int], top_k: int) -> list[Candidate]:
        query = self.embedder.encode([question])[0]
        k = min(top_k, len(allowed_indices))
        if len(allowed_indices) == len(self.chunks):
            scores, ids = self.index.search(query.reshape(1, -1), min(k, self.index.ntotal))
            pairs = [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx >= 0]
        else:
            matrix = self.embeddings[np.array(allowed_indices)]
            scores = matrix @ query
            order = np.argsort(-scores)[:k]
            pairs = [(allowed_indices[int(position)], float(scores[int(position)])) for position in order]
        return [
            self._candidate(index, vector_score=score, vector_rank=rank)
            for rank, (index, score) in enumerate(pairs[:k], start=1)
        ]

    def _bm25_search(self, question: str, allowed_indices: list[int], top_k: int) -> list[Candidate]:
        query_tokens = tokenize(question)
        scored = self._bm25.score(query_tokens, allowed_indices)
        return [
            self._candidate(index, bm25_score=score, bm25_rank=rank)
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]

    def _merge_candidates(self, *candidate_groups: list[Candidate]) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for group in candidate_groups:
            for candidate in group:
                chunk_id = str(candidate.metadata["chunk_id"])
                existing = merged.get(chunk_id)
                if existing is None:
                    merged[chunk_id] = candidate
                    continue
                if candidate.vector_score is not None:
                    existing.vector_score = candidate.vector_score
                    existing.vector_rank = candidate.vector_rank
                if candidate.bm25_score is not None:
                    existing.bm25_score = candidate.bm25_score
                    existing.bm25_rank = candidate.bm25_rank
        return list(merged.values())

    def _fuse_candidates(self, candidates: list[Candidate]) -> None:
        """Assign retrieval fusion scores before semantic reranking."""
        if self.fusion_method == "dbsf":
            vector_scores = distribution_based_scores([candidate.vector_score for candidate in candidates])
            bm25_scores = distribution_based_scores([candidate.bm25_score for candidate in candidates])
            for index, candidate in enumerate(candidates):
                candidate.fused_score = 0.5 * vector_scores[index] + 0.5 * bm25_scores[index]
            return

        if self.fusion_method != "rrf":
            raise ValueError("fusion_method must be 'rrf' or 'dbsf'.")

        for candidate in candidates:
            candidate.fused_score = reciprocal_rank_fusion_score(candidate.vector_rank, self.rrf_k)
            candidate.fused_score += reciprocal_rank_fusion_score(candidate.bm25_rank, self.rrf_k)

    def _rerank(self, question: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        if not candidates:
            return []

        candidates.sort(
            key=lambda candidate: (
                -candidate.fused_score,
                str(candidate.metadata["source_path"]),
                str(candidate.metadata["chunk_id"]),
            )
        )

        reranker_config = self.config.get("reranker", {})
        rerank_candidate_k = int(reranker_config.get("candidate_top_k", len(candidates)))
        rerank_pool = candidates[: max(top_k, min(rerank_candidate_k, len(candidates)))]
        cross_encoder_scores = self.reranker.score(question, rerank_pool)
        if cross_encoder_scores is not None:
            for candidate, score in zip(rerank_pool, cross_encoder_scores):
                candidate.cross_encoder_score = score

        cross_encoder_norm = normalize_scores([candidate.cross_encoder_score for candidate in candidates])
        fused_norm = normalize_scores([candidate.fused_score for candidate in candidates])
        question_terms = set(tokenize(question, drop_stopwords=True))
        has_cross_encoder = any(candidate.cross_encoder_score is not None for candidate in candidates)

        for index, candidate in enumerate(candidates):
            text_terms = set(tokenize(candidate.text, drop_stopwords=True))
            candidate.keyword_score = keyword_overlap(question_terms, text_terms)
            if has_cross_encoder:
                candidate.rerank_score = (
                    0.70 * cross_encoder_norm[index] + 0.20 * fused_norm[index] + 0.10 * candidate.keyword_score
                )
            else:
                candidate.rerank_score = 0.80 * fused_norm[index] + 0.20 * candidate.keyword_score

        candidates.sort(
            key=lambda candidate: (
                -candidate.rerank_score,
                -candidate.fused_score,
                str(candidate.metadata["source_path"]),
                str(candidate.metadata["chunk_id"]),
            )
        )
        return candidates[:top_k]

    def _candidate(
        self,
        index: int,
        vector_score: float | None = None,
        bm25_score: float | None = None,
        vector_rank: int | None = None,
        bm25_rank: int | None = None,
    ) -> Candidate:
        record = self.chunks[index]
        return Candidate(
            text=record["text"],
            metadata=record["metadata"],
            vector_score=vector_score,
            bm25_score=bm25_score,
            vector_rank=vector_rank,
            bm25_rank=bm25_rank,
        )


class CrossEncoderReranker:
    """Optional cross-encoder reranker with deterministic fallback."""

    def __init__(self, config: dict[str, Any]):
        reranker_config = config.get("reranker", {})
        self.backend = str(reranker_config.get("backend", "cross_encoder")).lower()
        self.model_name = reranker_config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.allow_fallback = bool(reranker_config.get("allow_fallback", True))
        self.actual_backend = "disabled"
        self._model = None

        if self.backend in {"none", "disabled", "off"}:
            return
        if self.backend != "cross_encoder":
            raise ValueError("reranker.backend must be 'cross_encoder' or 'none'.")

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self.actual_backend = "cross_encoder"
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(
                    "Could not load sentence-transformers CrossEncoder reranker. "
                    "Install requirements or set reranker.allow_fallback=true."
                ) from exc
            self.actual_backend = "deterministic_fallback"

    def score(self, question: str, candidates: list[Candidate]) -> list[float] | None:
        """Return cross-encoder scores when a model is available."""
        if self._model is None or not candidates:
            return None
        pairs = [(question, candidate.text) for candidate in candidates]
        scores = self._model.predict(pairs)
        return [float(score) for score in np.asarray(scores).reshape(-1)]


class BM25Index:
    """Small deterministic BM25 implementation for local keyword retrieval."""

    def __init__(self, tokenized_documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_documents = tokenized_documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(tokens) for tokens in tokenized_documents]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))
        total_documents = len(tokenized_documents)
        self.idf = {
            term: math.log(1 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.term_frequencies = [Counter(tokens) for tokens in tokenized_documents]

    def score(self, query_tokens: list[str], allowed_indices: list[int]) -> list[tuple[int, float]]:
        query_terms = [token for token in query_tokens if token in self.idf]
        if not query_terms:
            return []

        scored: list[tuple[int, float]] = []
        for index in allowed_indices:
            frequencies = self.term_frequencies[index]
            doc_length = self.doc_lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += self.idf[term] * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((index, float(score)))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored


def tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(str(text))]
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens


def normalize_scores(scores: list[float | None]) -> list[float]:
    """Min-max normalize present scores, treating missing scores as zero."""
    present = [float(score) for score in scores if score is not None]
    if not present:
        return [0.0 for _ in scores]
    minimum = min(present)
    maximum = max(present)
    if math.isclose(minimum, maximum):
        return [1.0 if score is not None else 0.0 for score in scores]
    return [0.0 if score is None else (float(score) - minimum) / (maximum - minimum) for score in scores]


def reciprocal_rank_fusion_score(rank: int | None, k: int = 60) -> float:
    """Return a standard reciprocal-rank fusion contribution."""
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def distribution_based_scores(scores: list[float | None]) -> list[float]:
    """Normalize scores with a distribution-based clamp before min-max scaling."""
    present = np.asarray([float(score) for score in scores if score is not None], dtype="float32")
    if present.size == 0:
        return [0.0 for _ in scores]
    mean = float(present.mean())
    std = float(present.std())
    if math.isclose(std, 0.0):
        return [1.0 if score is not None else 0.0 for score in scores]
    lower = mean - 3 * std
    upper = mean + 3 * std
    clipped = [None if score is None else min(max(float(score), lower), upper) for score in scores]
    return normalize_scores(clipped)


def keyword_overlap(question_terms: set[str], text_terms: set[str]) -> float:
    """Return simple question-term coverage in a chunk."""
    if not question_terms:
        return 0.0
    return len(question_terms & text_terms) / len(question_terms)
