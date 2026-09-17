"""The single budgeted scoring path for Framework V2 attacks.

Why this module exists
----------------------
The developmental Framework V2 benchmark called ``predictor.predict`` from at least
four independent places (the objective, the greedy loop, the benchmark script, and
drift recomputation), so its declared query budget ``Q`` did not bound anything:
482/540 runs asked the model more times than ``Q``
(docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 12).

Everything now goes through :class:`AttackEvaluator`:

* the source is scored **once**, before the search, and is NOT charged to ``Q``;
* every *new* candidate costs exactly **one** query and **one** model call;
* a candidate already seen (by :class:`~materials_adv.framework.interfaces.CandidateIdentity`)
  is answered from cache and costs nothing;
* scoring after exhaustion **raises** instead of silently continuing.

That makes ``predictor_predict_calls == 1 + queries_used`` an invariant the tests can
assert, for every strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from materials_adv.framework.budget import Budget
from materials_adv.framework.interfaces import Candidate, CandidateIdentity, Predictor, StringIdentity
from materials_adv.framework.objectives import AttackObjective


class BudgetExhausted(RuntimeError):
    """Raised when something tries to score after the query budget is spent."""


@dataclass
class ScoreRecord:
    index: int
    identity: str
    representation: Any
    prediction: float
    drift: float
    objective_value: float
    is_duplicate: bool
    operator_edits: Optional[int]
    provenance: List[str] = field(default_factory=list)


@dataclass
class RejectionRecord:
    identity: str
    reason: str


class AttackEvaluator:
    """Owns the only reference to the predictor used for candidate scoring."""

    def __init__(
        self,
        predictor: Predictor,
        objective: AttackObjective,
        budget: Budget,
        source: Candidate,
        identity: Optional[CandidateIdentity] = None,
    ) -> None:
        self._predictor = predictor
        self.objective = objective
        self.budget = budget
        self.source = source
        self._identity = identity or StringIdentity()

        self._cache: Dict[str, ScoreRecord] = {}
        self.trajectory: List[ScoreRecord] = []
        self.rejections: List[RejectionRecord] = []
        self.source_prediction: Optional[float] = None
        self.source_scorings = 0
        self.predictor_calls = 0
        self.duplicate_queries = 0

    # -- source ------------------------------------------------------------
    def prepare_source(self) -> float:
        """Score the source once, before the attack. Not charged to the budget."""
        if self.source_prediction is None:
            self.source_prediction = float(self._predictor.predict(self.source))
            self.predictor_calls += 1
            self.source_scorings += 1
        return self.source_prediction

    # -- identity / dedup --------------------------------------------------
    def identity_of(self, candidate: Candidate) -> str:
        return self._identity.identity(candidate)

    def is_duplicate(self, candidate: Candidate) -> bool:
        return self.identity_of(candidate) in self._cache

    def lookup(self, candidate: Candidate) -> Optional[ScoreRecord]:
        return self._cache.get(self.identity_of(candidate))

    def record_rejection(self, candidate: Candidate, reason: str) -> None:
        self.rejections.append(RejectionRecord(identity=self.identity_of(candidate), reason=reason))

    # -- the one scoring path ---------------------------------------------
    def score(self, candidate: Candidate, *, count_query: bool = True) -> ScoreRecord:
        if not count_query:
            raise ValueError(
                "unbudgeted scoring is not permitted: every candidate evaluation must "
                "be charged to the query budget (audit section 12)"
            )
        if self.source_prediction is None:
            raise RuntimeError("prepare_source() must be called before scoring candidates")

        key = self.identity_of(candidate)
        cached = self._cache.get(key)
        if cached is not None:
            # Answered from cache: no model call, no query spent, honest about it.
            self.duplicate_queries += 1
            return ScoreRecord(**{**cached.__dict__, "is_duplicate": True})

        if not self.budget.use_query():
            raise BudgetExhausted(
                f"query budget of {self.budget.max_queries} is exhausted; "
                "the evaluator refuses to score outside the budget"
            )

        prediction = float(self._predictor.predict(candidate))
        self.predictor_calls += 1
        drift = abs(prediction - self.source_prediction)
        record = ScoreRecord(
            index=self.budget.queries_used,
            identity=key,
            representation=candidate.identifier,
            prediction=prediction,
            drift=drift,
            objective_value=float(self.objective.score(self.source_prediction, prediction)),
            is_duplicate=False,
            operator_edits=candidate.operator_edits,
            provenance=list(candidate.provenance),
        )
        self._cache[key] = record
        self.trajectory.append(record)
        return record

    # -- introspection -----------------------------------------------------
    @property
    def queries_used(self) -> int:
        return self.budget.queries_used

    @property
    def remaining(self) -> int:
        return self.budget.remaining

    @property
    def unique_candidates(self) -> int:
        return len(self._cache)

    def best(self) -> Optional[ScoreRecord]:
        if not self.trajectory:
            return None
        return max(self.trajectory, key=lambda record: record.objective_value)

    def invariants(self) -> dict:
        return {
            "queries_used": self.budget.queries_used,
            "predictor_calls": self.predictor_calls,
            "source_scorings": self.source_scorings,
            "within_budget": self.budget.queries_used <= self.budget.max_queries,
            "one_model_call_per_query": self.predictor_calls == self.source_scorings + self.budget.queries_used,
        }
