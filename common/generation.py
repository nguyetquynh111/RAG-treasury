"""Shared answer generation helpers for vector RAG pipelines."""

from __future__ import annotations

import re
from typing import Any, Callable

from common.llm import GroundedRAGAnswerGenerator, SourceSnippet
from common.source_snippets import build_source_snippets as build_common_source_snippets


DEFAULT_MAX_CONTEXT_CHARS = 10_000
BASE_CITATION_FIELDS = ("source_path", "year_month", "chunk_id")
METADATA_CITATION_FIELDS = ("source_path", "year_month", "heading", "chunk_id")


class VectorRAGAnswerGenerator(GroundedRAGAnswerGenerator):
    """Generate answers from FAISS-retrieved chunk dictionaries."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        extractive_fallback: Callable[[str, list[dict[str, Any]]], str],
        citation_fields: tuple[str, ...] = BASE_CITATION_FIELDS,
        default_max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        self.citation_fields = citation_fields
        super().__init__(
            config,
            extractive_fallback=extractive_fallback,
            source_snippet_builder=self.build_source_snippets,
            default_max_context_chars=default_max_context_chars,
        )

    def build_source_snippets(
        self,
        chunks: list[dict[str, Any]],
        max_context_chars: int,
    ) -> list[SourceSnippet]:
        """Convert retrieved chunks into bounded source snippets."""
        return build_source_snippets(chunks, max_context_chars, self.citation_fields)


def build_source_snippets(
    chunks: list[dict[str, Any]],
    max_context_chars: int,
    citation_fields: tuple[str, ...] = BASE_CITATION_FIELDS,
) -> list[SourceSnippet]:
    """Convert retrieved chunk dictionaries into bounded source snippets."""
    return build_common_source_snippets(
        chunks,
        max_context_chars,
        get_text=lambda chunk: chunk.get("text", ""),
        get_metadata=lambda chunk: chunk.get("metadata", {}),
        citation_fields=citation_fields,
    )


def extractive_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    max_sentences: int = 2,
) -> str:
    """Return a cited fallback answer using sentences from retrieved chunks only."""
    if not retrieved_chunks:
        return "NOT_FOUND"

    question_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9]+", question)
        if len(term) > 2
    }
    candidates: list[tuple[int, int, str]] = []
    for source_index, chunk in enumerate(retrieved_chunks, start=1):
        sentences = re.split(r"(?<=[.!?])\s+|\n+", str(chunk.get("text", "")))
        for sentence in sentences:
            clean = " ".join(sentence.split())
            if not clean:
                continue
            sentence_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", clean)}
            overlap = len(question_terms & sentence_terms)
            candidates.append((overlap, source_index, clean))

    if not candidates:
        snippet = " ".join(str(retrieved_chunks[0].get("text", "")).split()[:80])
        return f"{snippet} [S1]" if snippet else "NOT_FOUND"

    candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    selected = [f"{sentence} [S{source_index}]" for _, source_index, sentence in candidates[:max_sentences]]
    return " ".join(selected) if selected else "NOT_FOUND"
