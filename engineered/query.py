"""Query planning for metadata-aware retrieval."""

from __future__ import annotations

from common.dataset import OfficeQARow
from common.query import QueryFilters, parse_query_filters


def filters_for_row(row: OfficeQARow, selected_years: list[int]) -> QueryFilters:
    """Prefer answer-key source dates, then fall back to dates mentioned in the question."""
    detected = parse_query_filters(row.question, selected_years)
    return QueryFilters(
        year=row.row_year if row.row_year is not None else detected.year,
        month=row.row_month if row.row_month is not None else None,
        date_pairs=tuple(row.source_date_pairs),
    )
