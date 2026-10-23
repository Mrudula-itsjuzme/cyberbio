from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class QueryBudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class QueryLedger:
    budget: int
    target_model_queries: int = 0
    proposal_count: int = 0
    validity_checks: int = 0
    attribution_queries: int = 0
    surrogate_queries: int = 0

    def charge_target(self, count: int = 1) -> None:
        if count < 0 or self.target_model_queries + count > self.budget:
            raise QueryBudgetExceeded(
                f"target-model budget {self.budget} would be exceeded"
            )
        self.target_model_queries += count

    @property
    def remaining_target(self) -> int:
        return self.budget - self.target_model_queries


class CountedTargetModel:
    """The only target-model gateway allowed in lab search implementations."""

    def __init__(self, model: Any, ledger: QueryLedger):
        self.model = model
        self.ledger = ledger
        self.history: list[tuple[str, float]] = []

    def predict(self, sequence: str) -> float:
        self.ledger.charge_target()
        value = float(self.model.predict(sequence))
        self.history.append((str(sequence), value))
        return value

    def predict_batch(self, sequences: Iterable[str]) -> list[float]:
        values = list(sequences)
        self.ledger.charge_target(len(values))
        if hasattr(self.model, "predict_batch"):
            predictions = [float(v) for v in self.model.predict_batch(values)]
        else:
            predictions = [float(self.model.predict(v)) for v in values]
        self.history.extend(zip(map(str, values), predictions))
        return predictions

    def last_prediction(self, sequence: str) -> float | None:
        for seen, prediction in reversed(self.history):
            if seen == sequence:
                return prediction
        return None
