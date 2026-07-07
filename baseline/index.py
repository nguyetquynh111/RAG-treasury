"""Build and save the baseline FAISS vector index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baseline.chunking import chunk_documents
from baseline.config import DEFAULT_CONFIG_PATH, load_config, resolve_path
from baseline.dataset import load_treasury_documents
from common.embeddings import Embedder
from common.chunk_io import save_chunks
from common.vector_index import build_inner_product_index, save_vector_index


def build_index(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, str | int]:
    """Load data, create chunks, embed them, and save a FAISS index locally."""
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
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "index_path": str(index_path),
        "chunks_path": str(chunks_path),
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the baseline Treasury FAISS index.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to baseline YAML config.")
    args = parser.parse_args()

    result = build_index(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
