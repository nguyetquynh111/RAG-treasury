"""Build the single shared Treasury RAG index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from common.chunking import chunk_documents, full_chunk_metadata, save_chunks
from common.config import DEFAULT_CONFIG_PATH, load_index_config, resolve_path
from common.dataset import load_treasury_documents
from common.embeddings import Embedder
from common.vector_store import build_inner_product_index, save_vector_index


PathLike = Union[str, Path]
ConfigLoader = Callable[[PathLike], dict[str, Any]]
PathResolver = Callable[[PathLike, PathLike], Path]
DocumentLoader = Callable[[PathLike, list[int]], list[Any]]


def build_index(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, str | int]:
    """Build chunks, embeddings, and the shared vector index from config."""
    return build_shared_index(
        config_path,
        load_config=load_index_config,
        resolve_path=resolve_path,
        load_documents=load_treasury_documents,
    )


def build_shared_index(
    config_path: str | Path,
    *,
    load_config: ConfigLoader,
    resolve_path: PathResolver,
    load_documents: DocumentLoader,
) -> dict[str, str | int]:
    """Build the shared index artifacts once for all RAG modes."""
    config = load_config(config_path)
    data_dir = resolve_path(config_path, config["data_dir"])
    index_dir = resolve_path(config_path, config["index_dir"])
    index_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(data_dir, config["document_years"])
    chunks = chunk_documents(
        documents,
        config["chunk_size"],
        config["chunk_overlap"],
        build_metadata=full_chunk_metadata,
    )

    embedder = Embedder(config)
    vectors = embedder.encode([chunk.text for chunk in chunks])
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {vectors.shape}")

    index_path = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "manifest.json"

    index = build_inner_product_index(vectors)
    index_backend = save_vector_index(index, vectors, index_path)
    save_chunks(chunks, chunks_path)
    np.save(embeddings_path, vectors)

    manifest = {
        "vector_db_type": "faiss",
        "index_backend": index_backend,
        "shared_index": True,
        "metadata_scope": "superset",
        "metadata_filtering": "retrieval_time_only",
        "embedding_backend": embedder.actual_backend,
        "embedding_model_name": embedder.model_name,
        "embedding_dim": int(vectors.shape[1]),
        "normalize_embeddings": bool(config.get("embedding", {}).get("normalize", True)),
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "selected_years": config["selected_years"],
        "document_years": config["document_years"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "top_k": config["top_k"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "index_path": str(index_path),
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    """CLI entrypoint for building the shared index."""
    parser = argparse.ArgumentParser(description="Build the shared Treasury FAISS index.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to shared YAML config.")
    args = parser.parse_args()

    print(json.dumps(build_index(args.config), indent=2))


if __name__ == "__main__":
    main()
