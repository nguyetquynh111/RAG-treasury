"""Shared interfaces for QA mode-specific behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from common.dataset import OfficeQARow
from common.retrieval import VectorRetriever


@dataclass(frozen=True)
class RetrievalPlan:
    """Metadata plan produced by one QA mode for one question."""

    year: int | None
    month: int | None
    date_pairs: tuple[tuple[int, int], ...] = ()


class ModeBehavior(Protocol):
    """Protocol implemented by baseline.retrieval and engineered.retrieval."""

    RETRIEVAL_METHOD: str

    def build_retrieval_plan(self, row: OfficeQARow, config: dict[str, Any]) -> RetrievalPlan:
        """Build the retrieval plan for one QA row."""

    def retrieve(
        self,
        retriever: VectorRetriever,
        question: str,
        config: dict[str, Any],
        plan: RetrievalPlan,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks for one question."""

    def extra_log_fields(self, plan: RetrievalPlan) -> dict[str, Any]:
        """Return mode-specific retrieval log fields."""
