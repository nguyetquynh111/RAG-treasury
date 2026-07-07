"""Engineered RAG answer generation adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from common.rag_generation import (
    DEFAULT_GENERATION_MODEL,
    GroundedRAGAnswerGenerator,
    SourceSnippet,
    build_messages,
)
from common.source_snippets import build_source_snippets as build_common_source_snippets

if TYPE_CHECKING:
    from engineered.retrieval import Candidate


DEFAULT_MAX_CONTEXT_CHARS = 12_000
CITATION_FIELDS = ("source_path", "year_month", "heading", "chunk_id")


class RAGAnswerGenerator(GroundedRAGAnswerGenerator):
    """Generate engineered answers from hybrid-retrieved candidates."""

    def __init__(
        self,
        config: dict[str, Any],
        extractive_fallback: Callable[[str, list[Candidate]], str],
    ) -> None:
        super().__init__(
            config,
            extractive_fallback=extractive_fallback,
            source_snippet_builder=build_source_snippets,
            default_max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
        )


def build_source_snippets(candidates: list[Candidate], max_context_chars: int) -> list[SourceSnippet]:
    """Convert engineered retrieved candidates into bounded source snippets."""
    return build_common_source_snippets(
        candidates,
        max_context_chars,
        get_text=lambda candidate: candidate.text,
        get_metadata=lambda candidate: candidate.metadata,
        citation_fields=CITATION_FIELDS,
    )


__all__ = ["DEFAULT_GENERATION_MODEL", "RAGAnswerGenerator", "build_messages", "build_source_snippets"]
