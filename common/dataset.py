"""Shared data-loading helpers for Treasury RAG pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
YEAR_PATTERN = r"(?:19|20)\d{2}"
QUESTION_COLUMNS = ("question", "query", "prompt")
ANSWER_COLUMNS = ("answer", "gold_answer", "ground_truth", "target")
ID_COLUMNS = ("question_id", "id", "uid", "qid", "row_id")
YEAR_COLUMNS = ("answer_year", "year", "document_year", "source_year")
MONTH_COLUMNS = ("answer_month", "month", "document_month", "source_month")
SOURCE_COLUMNS = {
    "source",
    "sources",
    "source_doc",
    "source_docs",
    "source_file",
    "source_files",
    "source_path",
    "source_paths",
    "document",
    "document_id",
    "doc_id",
    "filename",
    "file",
    "path",
    "url",
    "date",
    "period",
    "answer_source",
}


@dataclass(frozen=True)
class TreasuryDocument:
    text: str
    year: int
    month: int
    source_path: str


@dataclass(frozen=True)
class OfficeQARow:
    question_id: str
    question: str
    gold_answer: str
    row_year: int
    row_month: int | None


def extract_year_month(value: str) -> tuple[int, int] | None:
    """Extract the first explicit year/month pair from filenames, paths, dates, or text."""
    pairs = extract_all_year_month(value)
    return pairs[0] if pairs else None


def extract_all_year_month(value: str) -> list[tuple[int, int]]:
    """Extract all explicit year/month pairs from a metadata field."""
    text = str(value)
    lowered = text.lower()
    pairs: list[tuple[int, int]] = []

    numeric_patterns = [
        rf"(?P<year>{YEAR_PATTERN})[_\-/\.](?P<month>1[0-2]|0?[1-9])",
        rf"(?P<month>1[0-2]|0?[1-9])[_\-/\.](?P<year>{YEAR_PATTERN})",
    ]
    for pattern in numeric_patterns:
        for match in re.finditer(pattern, text):
            pairs.append((int(match.group("year")), int(match.group("month"))))

    for name, month in MONTH_NAMES.items():
        for match in re.finditer(rf"\b{name}\b[^\n\r,;|]*?\b(?P<year>{YEAR_PATTERN})\b", lowered):
            pairs.append((int(match.group("year")), month))
        for match in re.finditer(rf"\b(?P<year>{YEAR_PATTERN})\b[^\n\r,;|]*?\b{name}\b", lowered):
            pairs.append((int(match.group("year")), month))

    return dedupe_preserving_order(pairs)


def load_treasury_documents(data_dir: str | Path, selected_years: Iterable[int]) -> list[TreasuryDocument]:
    """Load selected Treasury text files and require year/month metadata."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Treasury data directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Treasury data path is not a directory: {root}")

    selected = {int(year) for year in selected_years}
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found under Treasury data directory: {root}")

    documents: list[TreasuryDocument] = []
    for path in files:
        extracted = extract_year_month(str(path))
        if extracted is None:
            raise ValueError(
                f"Could not extract year/month from Treasury file path: {path}. "
                "Expected a pattern like 2024_09 or 'September 2024'."
            )
        year, month = extracted
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month {month} extracted from Treasury file path: {path}")
        if year not in selected:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise ValueError(f"Treasury text file is empty: {path}")
        documents.append(TreasuryDocument(text=text, year=year, month=month, source_path=str(path)))

    if not documents:
        raise ValueError(f"No Treasury files found for selected years: {sorted(selected)}")
    return documents


def load_officeqa_rows(csv_path: str | Path) -> list[OfficeQARow]:
    """Load every officeqa_full.csv row with normalized question/answer/year metadata."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"OfficeQA CSV not found: {path}. Place officeqa_full.csv there or update csv_path in config."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"OfficeQA CSV is empty: {path}")

    columns = list(df.columns)
    question_col = find_column(columns, QUESTION_COLUMNS)
    answer_col = find_column(columns, ANSWER_COLUMNS)
    id_col = find_column(columns, ID_COLUMNS)
    if question_col is None:
        raise ValueError(f"Could not find a question column in {path}. Columns: {columns}")
    if answer_col is None:
        raise ValueError(f"Could not find an answer column in {path}. Columns: {columns}")

    rows: list[OfficeQARow] = []
    for index, row in df.iterrows():
        row_year, row_month = derive_row_year_month(row, columns, question_col)
        question_id = str(row[id_col]) if id_col is not None and not pd.isna(row[id_col]) else str(index)
        rows.append(
            OfficeQARow(
                question_id=question_id,
                question=str(row[question_col]),
                gold_answer=str(row[answer_col]),
                row_year=row_year,
                row_month=row_month,
            )
        )
    return rows


def load_filtered_officeqa_rows(csv_path: str | Path, selected_years: Iterable[int]) -> list[OfficeQARow]:
    """Load OfficeQA rows and keep only rows associated with selected years."""
    selected = {int(year) for year in selected_years}
    rows = [row for row in load_officeqa_rows(csv_path) if row.row_year in selected]
    if not rows:
        raise ValueError(f"No OfficeQA rows remain after filtering {csv_path} to selected years {sorted(selected)}.")
    return rows


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Find a case-insensitive column match."""
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def derive_row_year_month(row: pd.Series, columns: list[str], question_col: str) -> tuple[int, int | None]:
    """Derive answer source year/month from explicit metadata, source fields, or question text."""
    explicit = explicit_year_month(row, columns)
    if explicit is not None:
        return explicit

    source_pairs = source_year_month_pairs(row, columns)
    if source_pairs:
        return summarize_year_month_pairs(source_pairs)

    question_pairs = extract_all_year_month(str(row[question_col]))
    if question_pairs:
        return summarize_year_month_pairs(question_pairs)

    year_matches = [int(year) for year in re.findall(rf"\b({YEAR_PATTERN})\b", str(row[question_col]))]
    if year_matches:
        return min(set(year_matches)), None

    raise ValueError(
        "Could not determine the OfficeQA row year. Add a year/source/date column or include a year in the question."
    )


def explicit_year_month(row: pd.Series, columns: list[str]) -> tuple[int, int | None] | None:
    """Read explicit year/month columns when present."""
    year_col = find_column(columns, YEAR_COLUMNS)
    month_col = find_column(columns, MONTH_COLUMNS)

    year: int | None = None
    month: int | None = None
    if year_col is not None and not pd.isna(row[year_col]):
        year = int(row[year_col])
    if month_col is not None and not pd.isna(row[month_col]):
        month = int(row[month_col])

    if year is None:
        return None
    if month is not None and not 1 <= month <= 12:
        raise ValueError(f"Invalid OfficeQA month {month} in row {getattr(row, 'name', '<unknown>')}")
    return year, month


def source_year_month_pairs(row: pd.Series, columns: list[str]) -> list[tuple[int, int]]:
    """Collect year/month pairs from source-like metadata columns."""
    pairs: list[tuple[int, int]] = []
    for column in columns:
        if column.lower().strip() in SOURCE_COLUMNS and not pd.isna(row[column]):
            pairs.extend(extract_all_year_month(str(row[column])))
    return pairs


def summarize_year_month_pairs(pairs: list[tuple[int, int]]) -> tuple[int, int | None]:
    """Summarize one or more source dates into a row year and optional month."""
    years = {year for year, _ in pairs}
    months = {month for _, month in pairs}
    return max(years), months.pop() if len(months) == 1 else None


def dedupe_preserving_order(values: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Deduplicate tuple values while preserving first occurrence order."""
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result
