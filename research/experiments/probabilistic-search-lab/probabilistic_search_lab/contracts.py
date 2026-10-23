from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class MethodStatus(StrEnum):
    WORKS = "WORKS"
    BLOCKED_DEPENDENCY = "BLOCKED_DEPENDENCY"
    BLOCKED_COMPUTE = "BLOCKED_COMPUTE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    FAILED = "FAILED"


@dataclass(slots=True)
class SearchMetrics:
    proposal_count: int = 0
    validity_checks: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    target_model_queries: int = 0
    attribution_queries: int = 0
    surrogate_queries: int = 0
    runtime_seconds: float | None = None
    peak_memory_mb: float | None = None
    mean_effect: float | None = None
    median_effect: float | None = None
    p90_effect: float | None = None
    best_effect: float | None = None
    diversity_score: float | None = None


@dataclass(slots=True)
class SearchResult:
    method: str
    status: MethodStatus
    seed: int
    source_id: str
    query_budget: int
    metrics: SearchMetrics = field(default_factory=SearchMetrics)
    best_candidate: str | None = None
    failure_status: str | None = None
    trajectories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value
