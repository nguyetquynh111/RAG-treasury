"""Baseline retrieval behavior."""

from __future__ import annotations

from typing import Any

from common.dataset import OfficeQARow
from common.mode import RetrievalPlan
from common.retrieval import VectorRetriever


RETRIEVAL_METHOD = "faiss_vector_top_k_rag_generate"


def build_retrieval_plan(row: OfficeQARow, config: dict[str, Any]) -> RetrievalPlan:
    """Keep row date metadata for reporting, but do not filter retrieval."""
    return RetrievalPlan(year=row.row_year, month=row.row_month)


def retrieve(
    retriever: VectorRetriever,
    question: str,
    config: dict[str, Any],
    plan: RetrievalPlan,
) -> list[dict[str, Any]]:
    """Retrieve vector top-k results from the full shared index."""
    return retriever.retrieve(question, top_k=config["top_k"])


def extra_log_fields(plan: RetrievalPlan) -> dict[str, Any]:
    """Baseline adds no mode-specific log fields."""
    return {}
