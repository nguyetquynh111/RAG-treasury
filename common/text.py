"""Shared text helpers for Treasury RAG pipelines."""

from __future__ import annotations

from typing import Any


def split_tokens(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into whitespace-token windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start += step
    return chunks


def normalize_text(value: Any) -> str:
    """Collapse whitespace and coerce a value into a string."""
    return " ".join(str(value).split())
