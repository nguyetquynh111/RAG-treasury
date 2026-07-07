"""Shared text helpers for Treasury RAG pipelines."""

from __future__ import annotations

from typing import Any


def split_tokens(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into whitespace-token windows."""
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
