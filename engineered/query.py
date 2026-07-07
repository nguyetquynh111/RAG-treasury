"""Metadata-aware question parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.dataset import MONTH_NAMES


@dataclass(frozen=True)
class QueryFilters:
    year: int | None
    month: int | None


def parse_query_filters(question: str, selected_years: list[int]) -> QueryFilters:
    """Detect year and month filters in an OfficeQA question."""
    text = str(question)
    lowered = text.lower()
    selected = {int(year) for year in selected_years}

    for pattern in [
        r"\b(?P<month>1[0-2]|0?[1-9])[/\-.](?P<year>20\d{2})\b",
        r"\b(?P<year>20\d{2})[/\-.](?P<month>1[0-2]|0?[1-9])\b",
    ]:
        match = re.search(pattern, text)
        if match:
            year = int(match.group("year"))
            month = int(match.group("month"))
            return QueryFilters(year=year if year in selected else year, month=month)

    year_matches = [int(value) for value in re.findall(r"\b(20\d{2})\b", text)]
    detected_year = next((year for year in year_matches if year in selected), year_matches[0] if year_matches else None)

    detected_month: int | None = None
    for name, month in MONTH_NAMES.items():
        if re.search(rf"\b{name}\b", lowered):
            detected_month = month
            break

    if detected_year is None:
        return QueryFilters(year=None, month=None)
    return QueryFilters(year=detected_year, month=detected_month)
