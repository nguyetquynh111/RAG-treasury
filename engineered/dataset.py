"""Data loading for the engineered Treasury RAG system."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from common.dataset import (
    OfficeQARow,
    TreasuryDocument,
    extract_all_year_month,
    extract_year_month,
    load_filtered_officeqa_rows,
    load_officeqa_rows,
    load_treasury_documents,
)


def load_officeqa(csv_path: str | Path) -> list[OfficeQARow]:
    """Load every OfficeQA row with shared normalized metadata."""
    return load_officeqa_rows(csv_path)


def load_filtered_officeqa(csv_path: str | Path, selected_years: Iterable[int]) -> list[OfficeQARow]:
    """Load OfficeQA rows associated with selected years."""
    return load_filtered_officeqa_rows(csv_path, selected_years)
