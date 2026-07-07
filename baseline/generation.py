"""Baseline RAG answer generation adapter."""

from __future__ import annotations

from typing import Any, Callable

from common.rag_generation import GroundedRAGAnswerGenerator, SourceSnippet
from common.source_snippets import build_source_snippets as build_common_source_snippets


DEFAULT_MAX_CONTEXT_CHARS = 10_000
CITATION_FIELDS = ("source_path", "year_month", "chunk_id")


class BaselineRAGAnswerGenerator(GroundedRAGAnswerGenerator):
    """Generate baseline answers from FAISS-retrieved chunks."""

    def __init__(
        self,
        config: dict[str, Any],
        extractive_fallback: Callable[[str, list[dict[str, Any]]], str],
    ) -> None:
        super().__init__(
            config,
            extractive_fallback=extractive_fallback,
            source_snippet_builder=build_source_snippets,
            default_max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
        )


def build_source_snippets(chunks: list[dict[str, Any]], max_context_chars: int) -> list[SourceSnippet]:
    """Convert baseline retrieved chunks into bounded source snippets."""
    return build_common_source_snippets(
        chunks,
        max_context_chars,
        get_text=lambda chunk: chunk.get("text", ""),
        get_metadata=lambda chunk: chunk.get("metadata", {}),
        citation_fields=CITATION_FIELDS,
    )
