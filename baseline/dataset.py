"""Dataset and answer-key loading for the baseline Treasury RAG system."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from common.dataset import (
    OfficeQARow,
    TreasuryDocument,
    extract_all_year_month,
    extract_year_month,
    load_filtered_officeqa_rows,
    load_officeqa_rows,
    load_treasury_documents,
)


def load_officeqa(csv_path: str | Path) -> pd.DataFrame:
    """Load every OfficeQA row and expose baseline-compatible column names."""
    rows = load_officeqa_rows(csv_path)
    return officeqa_rows_to_baseline_frame(rows)


def load_filtered_officeqa(csv_path: str | Path, selected_years: Iterable[int]) -> pd.DataFrame:
    """Load OfficeQA rows in selected years as a baseline-compatible frame."""
    rows = load_filtered_officeqa_rows(csv_path, selected_years)
    return officeqa_rows_to_baseline_frame(rows)


def officeqa_rows_to_baseline_frame(rows: list[OfficeQARow]) -> pd.DataFrame:
    """Convert shared OfficeQA rows into the original baseline DataFrame schema."""
    records = [
        {
            "question_id": row.question_id,
            "baseline_question": row.question,
            "baseline_answer": row.gold_answer,
            "baseline_year": row.row_year,
            "baseline_month": row.row_month,
        }
        for row in rows
    ]
    return pd.DataFrame(records).reset_index(drop=True)
