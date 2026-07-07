"""Shared chunk data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    metadata: dict[str, int | str]
