"""Shared YAML configuration helpers for Treasury RAG pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


SUPPORTED_VECTOR_DB_TYPES = {"faiss"}
SUPPORTED_EMBEDDING_BACKENDS = {"deepinfra"}
SUPPORTED_GENERATION_BACKENDS = {"deepinfra", "openai", "extractive"}
EMBEDDING_POSITIVE_INT_KEYS = ("batch_size", "timeout_seconds")
EMBEDDING_NONNEGATIVE_INT_KEYS = ("max_retries",)
EMBEDDING_NONNEGATIVE_FLOAT_KEYS = ("request_sleep_seconds", "retry_sleep_seconds")
GENERATION_POSITIVE_INT_KEYS = ("timeout_seconds", "max_tokens", "max_context_chars", "num_ctx")
GENERATION_NONNEGATIVE_INT_KEYS = ("max_retries",)
GENERATION_NONNEGATIVE_FLOAT_KEYS = ("request_sleep_seconds", "retry_sleep_seconds")


def load_yaml_config(config_path: str | Path, *, label: str) -> dict[str, Any]:
    """Load one YAML config file and require a mapping at the root."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"{label} config must be a YAML mapping: {path}")
    return config


def require_keys(config: dict[str, Any], required_keys: Iterable[str], *, path: Path) -> None:
    """Fail loudly when a config omits required top-level keys."""
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys in {path}: {missing}")


def validate_faiss_config(config: dict[str, Any], *, label: str) -> None:
    """Require the currently supported vector database backend."""
    if config["vector_db_type"] not in SUPPORTED_VECTOR_DB_TYPES:
        raise ValueError(f"The {label} pipeline implements only vector_db_type='faiss'.")


def normalize_selected_years(config: dict[str, Any]) -> None:
    """Validate and normalize selected_years in place."""
    years = config["selected_years"]
    if not isinstance(years, list) or not years:
        raise ValueError("selected_years must be a non-empty list of years.")
    config["selected_years"] = [int(year) for year in years]


def normalize_chunk_settings(config: dict[str, Any]) -> None:
    """Validate chunk_size and chunk_overlap in place."""
    config["chunk_size"] = int(config["chunk_size"])
    config["chunk_overlap"] = int(config["chunk_overlap"])
    if config["chunk_size"] <= 0:
        raise ValueError("chunk_size must be positive.")
    if config["chunk_overlap"] < 0 or config["chunk_overlap"] >= config["chunk_size"]:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")


def normalize_positive_ints(config: dict[str, Any], keys: Iterable[str]) -> None:
    """Validate required positive integer settings in place."""
    for key in keys:
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive.")


def normalize_embedding_config(config: dict[str, Any]) -> None:
    """Validate optional embedding settings in place."""
    embedding = config.get("embedding", {})
    if embedding is None:
        embedding = {}
    if not isinstance(embedding, dict):
        raise ValueError("embedding must be a mapping when provided.")

    backend = str(embedding.get("backend", "deepinfra")).lower()
    if backend not in SUPPORTED_EMBEDDING_BACKENDS:
        allowed = " or ".join(f"'{name}'" for name in sorted(SUPPORTED_EMBEDDING_BACKENDS))
        raise ValueError(f"embedding.backend must be {allowed}.")
    embedding["backend"] = backend

    for key in EMBEDDING_POSITIVE_INT_KEYS:
        if key in embedding:
            embedding[key] = int(embedding[key])
            if embedding[key] <= 0:
                raise ValueError(f"embedding.{key} must be positive.")
    for key in EMBEDDING_NONNEGATIVE_INT_KEYS:
        if key in embedding:
            embedding[key] = int(embedding[key])
            if embedding[key] < 0:
                raise ValueError(f"embedding.{key} must be non-negative.")
    for key in EMBEDDING_NONNEGATIVE_FLOAT_KEYS:
        if key in embedding:
            embedding[key] = float(embedding[key])
            if embedding[key] < 0:
                raise ValueError(f"embedding.{key} must be non-negative.")

    config["embedding"] = embedding


def normalize_generation_config(config: dict[str, Any]) -> None:
    """Validate optional generation settings in place."""
    generation = config.get("generation", {})
    if generation is None:
        config["generation"] = {}
        return
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping when provided.")

    backend = str(generation.get("backend", "deepinfra")).lower()
    if backend not in SUPPORTED_GENERATION_BACKENDS:
        allowed = " or ".join(f"'{name}'" for name in sorted(SUPPORTED_GENERATION_BACKENDS))
        raise ValueError(f"generation.backend must be {allowed}.")
    generation["backend"] = backend

    for key in GENERATION_POSITIVE_INT_KEYS:
        if key in generation:
            generation[key] = int(generation[key])
            if generation[key] <= 0:
                raise ValueError(f"generation.{key} must be positive.")
    for key in GENERATION_NONNEGATIVE_INT_KEYS:
        if key in generation:
            generation[key] = int(generation[key])
            if generation[key] < 0:
                raise ValueError(f"generation.{key} must be non-negative.")
    for key in GENERATION_NONNEGATIVE_FLOAT_KEYS:
        if key in generation:
            generation[key] = float(generation[key])
            if generation[key] < 0:
                raise ValueError(f"generation.{key} must be non-negative.")
    if "temperature" in generation:
        generation["temperature"] = float(generation["temperature"])

    config["generation"] = generation


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return Path(config_path).resolve().parent.parent / candidate
