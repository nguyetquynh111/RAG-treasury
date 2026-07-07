"""Source-snippet adapters for grounded generation."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from common.llm import SourceSnippet
from common.text import normalize_text

TextGetter = Callable[[Any], Any]
MetadataGetter = Callable[[Any], dict[str, Any]]


def build_source_snippets(
    items: list[Any],
    max_context_chars: int,
    *,
    get_text: TextGetter,
    get_metadata: MetadataGetter,
    citation_fields: Iterable[str],
) -> list[SourceSnippet]:
    """Convert retrieved items into bounded source snippets."""
    snippets: list[SourceSnippet] = []
    remaining_chars = max(0, max_context_chars)

    for index, item in enumerate(items, start=1):
        if remaining_chars <= 0:
            break

        text = normalize_text(get_text(item))
        if not text:
            continue

        clipped_text = text[:remaining_chars]
        remaining_chars -= len(clipped_text)
        snippets.append(
            SourceSnippet(
                label=f"S{index}",
                text=clipped_text,
                citation=build_citation(get_metadata(item), citation_fields),
            )
        )
    return snippets


def build_citation(metadata: dict[str, Any], fields: Iterable[str]) -> str:
    """Return a compact source citation from selected metadata fields."""
    values: list[str] = []
    for field in fields:
        if field == "year_month":
            values.append(format_year_month(metadata))
        else:
            values.append(str(metadata.get(field, f"unknown {field}")))
    return " | ".join(values)


def format_year_month(metadata: dict[str, Any]) -> str:
    """Format year/month metadata for source citations."""
    year = metadata.get("year", "unknown year")
    month = metadata.get("month")
    month_text = f"{int(month):02d}" if month is not None else "unknown month"
    return f"{year}-{month_text}"
