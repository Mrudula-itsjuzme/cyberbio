"""Budgeted, domain-agnostic black-box search strategies.

Every strategy here:

* scores exclusively through :class:`~materials_adv.framework.accounting.AttackEvaluator`;
* never holds a predictor reference of its own, so it *cannot* score off-budget;
* knows nothing about SMILES, RDKit, atoms or chemistry -- it sees only
  ``Candidate`` / ``AttackOperator`` / ``ValidityChecker`` / ``CandidateIdentity``.

This module therefore contains no RDKit import, which is asserted by a regression
test (audit finding: the "generic" search layer used to pull RDKit in at import time
and defaulted to a chemistry validity checker).
"""

from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from materials_adv.framework.accounting import AttackEvaluator, ScoreRecord
from materials_adv.framework.interfaces import AttackOperator, Candidate, ValidityChecker


@dataclass
class SearchOutcome:
    strategy: str
    source_representation: Any
    best_representation: Any
    source_prediction: float
    best_prediction: float
    best_drift: float
    queries_used: int
    query_budget: int
    candidates_proposed: int = 0
    candidates_valid: int = 0
    candidates_unique: int = 0
    duplicates: int = 0
    invalid_proposals: int = 0
    edit_budget_rejections: int = 0
    best_operator_edits: Optional[int] = None
    max_operator_edits: Optional[int] = None
    mean_operator_edits: Optional[float] = None
    seed: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid_candidate_rate(self) -> float:
        return self.candidates_valid / self.candidates_proposed if self.candidates_proposed else 0.0

    @property
    def unique_candidate_rate(self) -> float:
        return self.candidates_unique / self.queries_used if self.queries_used else 0.0

    @property
    def query_efficiency(self) -> float:
        """Drift gained per candidate query spent."""
        return self.best_drift / self.queries_used if self.queries_used else 0.0

    def to_row(self) -> dict:
        return {
            "strategy": self.strategy,
            "query_budget": self.query_budget,
            "queries_used": self.queries_used,
            "candidates_proposed": self.candidates_proposed,
            "candidates_valid": self.candidates_valid,
            "valid_candidate_rate": self.valid_candidate_rate,
            "candidates_unique": self.candidates_unique,
            "unique_candidate_rate": self.unique_candidate_rate,
            "duplicates": self.duplicates,
            "invalid_proposals": self.invalid_proposals,
            "edit_budget_rejections": self.edit_budget_rejections,
            "source_prediction": self.source_prediction,
            "best_prediction": self.best_prediction,
            "best_drift": self.best_drift,
            "drift_per_query": self.query_efficiency,
            "best_operator_edits": self.best_operator_edits,
            "max_operator_edits": self.max_operator_edits,
            "mean_operator_edits": self.mean_operator_edits,
            "seed": self.seed,
        }


class BudgetedSearch(abc.ABC):
    """Common machinery: proposal filtering, dedup-before-spend, budget discipline."""

    name = "base"

    def __init__(
        self,
        operator: AttackOperator,
        evaluator: AttackEvaluator,
        validator: ValidityChecker,
        *,
        seed: Optional[int] = None,
        max_empty_rounds: int = 3,
    ) -> None:
        if not isinstance(evaluator, AttackEvaluator):
            raise TypeError(
                "search strategies must score through an AttackEvaluator; passing a raw "
                "predictor is what broke Framework V2's query accounting"
            )
        self.operator = operator
        self.evaluator = evaluator
        self.validator = validator
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_empty_rounds = max_empty_rounds

        self.candidates_proposed = 0
        self.candidates_valid = 0
        self.duplicates = 0
        self.invalid_proposals = 0
        self.edit_budget_rejections = 0
        self.empty_rounds = 0
        # A parent whose every child has been rejected or already scored can never
        # produce anything new, because the duplicate cache only grows. Remembering that
        # avoids re-enumerating an exhausted neighbourhood (which used to spin the
        # Metropolis and Evolutionary loops and inflate the proposal counters).
        # NOTE: assumes a deterministic operator, which is true of the chemistry ones.
        self.exhausted_parents: set[str] = set()

    def is_exhausted(self, candidate: Candidate) -> bool:
        return self.evaluator.identity_of(candidate) in self.exhausted_parents

    # -- proposal handling -------------------------------------------------
    def proposals(self, parent: Candidate) -> List[Candidate]:
        """Valid, within-edit-budget, not-yet-scored children of ``parent``."""
        parent_key = self.evaluator.identity_of(parent)
        if parent_key in self.exhausted_parents:
            return []
        raw = list(self.operator.apply(parent))
        self.candidates_proposed += len(raw)
        out: List[Candidate] = []
        seen = set()
        for candidate in raw:
            key = self.evaluator.identity_of(candidate)
            if key in seen:
                self.duplicates += 1
                continue
            seen.add(key)
            if not self.validator.is_valid(candidate):
                self.invalid_proposals += 1
                self.evaluator.record_rejection(candidate, "invalid_candidate")
                continue
            if not self.evaluator.budget.admits_edits(candidate):
                self.edit_budget_rejections += 1
                self.evaluator.record_rejection(candidate, "edit_budget_exceeded")
                continue
            if self.evaluator.is_duplicate(candidate):
                self.duplicates += 1
                continue
            out.append(candidate)
        self.candidates_valid += len(out)
        if not out:
            self.exhausted_parents.add(parent_key)
        return out

    # -- outcome assembly --------------------------------------------------
    def _outcome(self, source: Candidate, best: Optional[ScoreRecord]) -> SearchOutcome:
        source_prediction = float(self.evaluator.source_prediction)
        edits = [record.operator_edits for record in self.evaluator.trajectory
                 if record.operator_edits is not None]
        return SearchOutcome(
            strategy=self.name,
            source_representation=source.identifier,
            best_representation=best.representation if best else source.identifier,
            source_prediction=source_prediction,
            best_prediction=best.prediction if best else source_prediction,
            best_drift=best.drift if best else 0.0,
            queries_used=self.evaluator.queries_used,
            query_budget=self.evaluator.budget.max_queries,
            candidates_proposed=self.candidates_proposed,
            candidates_valid=self.candidates_valid,
            candidates_unique=self.evaluator.unique_candidates,
            duplicates=self.duplicates,
            invalid_proposals=self.invalid_proposals,
            edit_budget_rejections=self.edit_budget_rejections,
            best_operator_edits=best.operator_edits if best else None,
            max_operator_edits=max(edits) if edits else None,
            mean_operator_edits=(sum(edits) / len(edits)) if edits else None,
            seed=self.seed,
            config=self.config(),
        )

    def config(self) -> dict:
        return {"max_empty_rounds": self.max_empty_rounds}

    @abc.abstractmethod
    def search(self, source: Candidate) -> SearchOutcome:
        raise NotImplementedError


class RandomSearch(BudgetedSearch):
    """Uniform sampling from the source neighbourhood. No exploitation."""

    name = "random"

    def search(self, source: Candidate) -> SearchOutcome:
        self.evaluator.prepare_source()
        best: Optional[ScoreRecord] = None
        while not self.evaluator.budget.is_exhausted():
            pool = [c for c in self.proposals(source)
                    if not self.evaluator.is_duplicate(c)]
            if not pool:
                break
            candidate = self.rng.choice(pool)
            record = self.evaluator.score(candidate)
            if best is None or record.objective_value > best.objective_value:
                best = record
        return self._outcome(source, best)


class GreedySearch(BudgetedSearch):
    """Hill-climb: score the current neighbourhood, move to the best improvement."""

    name = "greedy"

    def search(self, source: Candidate) -> SearchOutcome:
        self.evaluator.prepare_source()
        current = source
        current_value = 0.0
        best: Optional[ScoreRecord] = None

        while not self.evaluator.budget.is_exhausted():
            pool = self.proposals(current)
            if not pool:
                break
            step_best: Optional[Tuple[Candidate, ScoreRecord]] = None
            for candidate in pool:
                if self.evaluator.budget.is_exhausted():
                    break
                record = self.evaluator.score(candidate)
                if step_best is None or record.objective_value > step_best[1].objective_value:
                    step_best = (candidate, record)
            if step_best is None:
                break
            candidate, record = step_best
            if best is None or record.objective_value > best.objective_value:
                best = record
            if record.objective_value > current_value:
                current, current_value = candidate, record.objective_value
            else:
                break
        return self._outcome(source, best)


class MetropolisSearch(BudgetedSearch):
    """Metropolis-style walk.

    Not claimed as formal Metropolis-Hastings: there is no proposal-distribution
    correction. ``temperature`` is in the model's own unit (eV for bandgaps).
    """

    name = "metropolis"

    def __init__(self, *args, temperature: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def config(self) -> dict:
        return {**super().config(), "temperature": self.temperature}

    def search(self, source: Candidate) -> SearchOutcome:
        self.evaluator.prepare_source()
        current = source
        current_value = 0.0
        best: Optional[ScoreRecord] = None

        while not self.evaluator.budget.is_exhausted():
            pool = self.proposals(current)
            if not pool:
                if current is not source and not self.is_exhausted(source):
                    # restart the walk from the source rather than stopping early
                    current, current_value = source, 0.0
                    continue
                break
            candidate = self.rng.choice(pool)
            record = self.evaluator.score(candidate)
            if best is None or record.objective_value > best.objective_value:
                best = record
            delta = record.objective_value - current_value
            if delta > 0 or self.rng.random() < math.exp(delta / self.temperature):
                current, current_value = candidate, record.objective_value
        return self._outcome(source, best)


class EvolutionarySearch(BudgetedSearch):
    """Population search with elitism.

    ``population_size`` is capped by the query budget, and both the requested and the
    effective value are reported -- a population larger than ``Q`` cannot fill a single
    generation, which would silently degenerate into a one-shot sampler.
    """

    name = "evolutionary"

    def __init__(self, *args, population_size: int = 8, elite_size: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if population_size < 1:
            raise ValueError("population_size must be positive")
        if not 1 <= elite_size <= population_size:
            raise ValueError("elite_size must satisfy 1 <= elite_size <= population_size")
        self.requested_population_size = population_size
        self.effective_population_size = min(population_size, self.evaluator.budget.max_queries)
        self.elite_size = min(elite_size, self.effective_population_size)

    def config(self) -> dict:
        return {
            **super().config(),
            "requested_population_size": self.requested_population_size,
            "effective_population_size": self.effective_population_size,
            "elite_size": self.elite_size,
            "population_capped_by_budget": self.effective_population_size != self.requested_population_size,
        }

    def search(self, source: Candidate) -> SearchOutcome:
        self.evaluator.prepare_source()
        population: List[Tuple[Candidate, float]] = [(source, 0.0)]
        best: Optional[ScoreRecord] = None
        consecutive_empty = 0

        while not self.evaluator.budget.is_exhausted():
            fertile = [(c, v) for c, v in population if not self.is_exhausted(c)]
            if not fertile:
                break
            offspring: List[Tuple[Candidate, float]] = []
            while len(offspring) < self.effective_population_size and not self.evaluator.budget.is_exhausted():
                parent, _ = self.rng.choice(fertile)
                pool = self.proposals(parent)
                if not pool:
                    consecutive_empty += 1
                    if consecutive_empty >= self.max_empty_rounds:
                        break
                    continue
                consecutive_empty = 0
                candidate = self.rng.choice(pool)
                record = self.evaluator.score(candidate)
                offspring.append((candidate, record.objective_value))
                if best is None or record.objective_value > best.objective_value:
                    best = record
            if not offspring:
                break
            pooled = {self.evaluator.identity_of(c): (c, v) for c, v in population + offspring}
            population = sorted(pooled.values(), key=lambda item: item[1], reverse=True)[: self.elite_size]
        return self._outcome(source, best)
