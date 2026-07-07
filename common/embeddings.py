"""Shared DeepInfra embedding helpers for Treasury RAG pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
from openai import OpenAI, RateLimitError

from common.rag_generation import (
    DEFAULT_DEEPINFRA_API_KEY_ENV,
    DEFAULT_OPENAI_BASE_URL,
    load_env_file,
    rate_limit_sleep_seconds,
    resolve_api_key,
    sleep_if_positive,
)


DEFAULT_EMBEDDING_BACKEND = "deepinfra"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 300
DEFAULT_EMBEDDING_MAX_RETRIES = 3
DEFAULT_EMBEDDING_RETRY_SLEEP_SECONDS = 5.0


class Embedder:
    """Encode text with DeepInfra's OpenAI-compatible embeddings API."""

    def __init__(self, config: dict[str, Any]):
        embedding_config = config.get("embedding", {})
        self.backend = str(embedding_config.get("backend", DEFAULT_EMBEDDING_BACKEND)).lower()
        if self.backend != "deepinfra":
            raise ValueError("embedding.backend must be 'deepinfra'.")

        self.model_name = str(embedding_config.get("model_name", DEFAULT_EMBEDDING_MODEL))
        self.base_url = str(embedding_config.get("base_url", DEFAULT_OPENAI_BASE_URL))
        self.api_key_env = str(
            embedding_config.get("api_key_env", DEFAULT_DEEPINFRA_API_KEY_ENV)
        )
        self.normalize = bool(embedding_config.get("normalize", True))
        self.batch_size = int(embedding_config.get("batch_size", DEFAULT_EMBEDDING_BATCH_SIZE))
        self.timeout_seconds = int(
            embedding_config.get("timeout_seconds", DEFAULT_EMBEDDING_TIMEOUT_SECONDS)
        )
        self.request_sleep_seconds = float(embedding_config.get("request_sleep_seconds", 0.0))
        self.retry_sleep_seconds = float(
            embedding_config.get("retry_sleep_seconds", DEFAULT_EMBEDDING_RETRY_SLEEP_SECONDS)
        )
        self.max_retries = int(embedding_config.get("max_retries", DEFAULT_EMBEDDING_MAX_RETRIES))
        self.actual_backend = self.backend

        if self.batch_size <= 0:
            raise ValueError("embedding.batch_size must be positive.")
        if self.timeout_seconds <= 0:
            raise ValueError("embedding.timeout_seconds must be positive.")
        if self.max_retries < 0:
            raise ValueError("embedding.max_retries must be non-negative.")
        if self.request_sleep_seconds < 0 or self.retry_sleep_seconds < 0:
            raise ValueError("embedding sleep settings must be non-negative.")

        self._client: OpenAI | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts as a 2D float32 matrix."""
        if not texts:
            raise ValueError("Cannot embed an empty text list.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
        if len(matrix) != len(texts):
            raise ValueError(f"Embedding count mismatch: got {len(matrix)} for {len(texts)} texts.")

        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix.astype("float32")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI-compatible embeddings endpoint with retry on rate limits."""
        client = self._openai_client()
        for attempt in range(self.max_retries + 1):
            sleep_if_positive(self.request_sleep_seconds)
            try:
                response = client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    timeout=self.timeout_seconds,
                )
                return [
                    list(item.embedding)
                    for item in sorted(response.data, key=lambda item: item.index)
                ]
            except RateLimitError as exc:
                if attempt == self.max_retries:
                    raise
                sleep_if_positive(
                    rate_limit_sleep_seconds(
                        getattr(exc, "response", None),
                        self.retry_sleep_seconds,
                    )
                )

        raise RuntimeError("Unexpected embedding retry loop exit.")

    def _openai_client(self) -> OpenAI:
        """Create the DeepInfra client lazily after loading .env."""
        if self._client is None:
            load_env_file()
            self._client = OpenAI(
                api_key=resolve_api_key(self.api_key_env),
                base_url=self.base_url,
            )
        return self._client
