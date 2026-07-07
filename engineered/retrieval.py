"""Engineered retrieval behavior."""

from __future__ import annotations

from typing import Any

from common.dataset import OfficeQARow
from common.mode import RetrievalPlan
from common.retrieval import VectorRetriever
from engineered.query import filters_for_row


RETRIEVAL_METHOD = "faiss_vector_top_k_metadata_rag_generate"


def build_retrieval_plan(row: OfficeQARow, config: dict[str, Any]) -> RetrievalPlan:
    """Build metadata filters from source dates and question dates."""
    filters = filters_for_row(row, config["selected_years"])
    return RetrievalPlan(filters.year, filters.month, filters.date_pairs)


def retrieve(
    retriever: VectorRetriever,
    question: str,
    config: dict[str, Any],
    plan: RetrievalPlan,
) -> list[dict[str, Any]]:
    """Retrieve vector top-k results after metadata filtering on the shared index."""
    return retriever.retrieve(
        question,
        top_k=config["top_k"],
        year=plan.year,
        month=plan.month,
        date_pairs=plan.date_pairs,
    )


def extra_log_fields(plan: RetrievalPlan) -> dict[str, Any]:
    """Add metadata filter provenance to engineered retrieval logs."""
    return {"source_date_pairs": list(plan.date_pairs)}
