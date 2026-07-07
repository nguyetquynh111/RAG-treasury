"""Configuration loading for the engineered Treasury RAG system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config import (
    load_yaml_config,
    normalize_chunk_settings,
    normalize_embedding_config,
    normalize_generation_config,
    normalize_positive_ints,
    normalize_selected_years,
    require_keys,
    resolve_path,
    validate_faiss_config,
)


DEFAULT_CONFIG_PATH = Path("config/engineered.yaml")
REQUIRED_CONFIG_KEYS = (
    "data_dir",
    "csv_path",
    "selected_years",
    "vector_db_type",
    "embedding",
    "chunk_size",
    "chunk_overlap",
    "bm25_top_k",
    "vector_top_k",
    "final_top_k",
    "output_dir",
)


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the engineered pipeline configuration."""
    path = Path(config_path)
    config = load_yaml_config(path, label="Engineered")
    require_keys(config, REQUIRED_CONFIG_KEYS, path=path)
    validate_faiss_config(config, label="engineered")
    normalize_selected_years(config)
    normalize_chunk_settings(config)
    normalize_positive_ints(config, ("bm25_top_k", "vector_top_k", "final_top_k"))
    normalize_embedding_config(config)
    normalize_generation_config(config)
    return config
