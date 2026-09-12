"""Black-box search policies with explicit model-query accounting."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from ...data.tokenizer import tokenize
from ...validation.pipeline import validate
from ..generator import PredictorProtocol
from ..token_space import count_changes
from .proposals import Proposal


class ProposalProtocol(Protocol):
    @property
    def names(self) -> tuple[str, ...]: ...
    def propose(self, tokens: Sequence[str]) -> Proposal | None: ...


@dataclass
class SearchResult:
    strategy: str
    original_representation: str
    best_representation: str
    original_prediction: float
    best_prediction: float
    best_drift: float
    candidate_query_budget: int
    candidate_model_queries: int
    total_model_queries: int
    candidate_proposals: int
    representation_valid_candidates: int
    valid_candidates: int
    accepted_candidates: int
    acceptance_rate: float | None
    perturbation_size: int
    elapsed_seconds: float
    proposal_operators: tuple[str, ...]
    trace: list[dict] = field(default_factory=list)

    @property
    def drift_per_model_query(self) -> float:
        return self.best_drift / self.candidate_model_queries if self.candidate_model_queries else 0.0

    def summary(self) -> dict:
        result = {key: value for key, value in self.__dict__.items() if key != "trace"}
        result["drift_per_model_query"] = self.drift_per_model_query
        result["query_budget_exhausted"] = self.candidate_model_queries == self.candidate_query_budget
        result["proposal_operators"] = list(self.proposal_operators)
        return result


@dataclass(frozen=True)
class _Candidate:
    tokens: tuple[str, ...]
    representation: str
    prediction: float
    drift: float


class BlackBoxSearch:
    name = "base"

    def __init__(self, predictor: PredictorProtocol, proposal: ProposalProtocol,
                 rng: np.random.Generator, *, query_budget: int,
                 require_plausible: bool = True, max_changes: int | None = None,
                 max_proposal_multiplier: int = 50):
        if query_budget < 1:
            raise ValueError("query_budget must be positive")
        self.predictor, self.proposal, self.rng = predictor, proposal, rng
        self.query_budget = query_budget
        self.require_plausible = require_plausible
        self.max_changes = max_changes
        self.max_proposals = query_budget * max_proposal_multiplier

    def _run(self, original: _Candidate, evaluate, record):
        raise NotImplementedError

    def search(self, representation: str) -> SearchResult:
        started = time.perf_counter()
        original_tokens = tuple(tokenize(representation))
        original_prediction = float(self.predictor.predict([representation])[0])
        original = _Candidate(original_tokens, representation, original_prediction, 0.0)
        cache = {representation: original}
        counters = {"proposals": 0, "rep_valid": 0, "valid": 0, "queries": 0, "accepted": 0}
        trace: list[dict] = []

        def evaluate(parent: _Candidate) -> tuple[_Candidate | None, Proposal | None, dict]:
            counters["proposals"] += 1
            proposed = self.proposal.propose(parent.tokens)
            entry = {"proposal_index": counters["proposals"], "parent": parent.representation,
                     "operator": None, "candidate": None, "representation_valid": False,
                     "plausible": False, "queried": False, "prediction": None,
                     "absolute_drift": None, "accepted": False}
            if proposed is None:
                entry["rejection_reason"] = "operator_returned_no_candidate"
                return None, None, entry
            text = "".join(proposed.tokens)
            perturbation_size = count_changes(list(original.tokens), list(proposed.tokens))
            entry.update({"operator": proposed.operator, "candidate": text,
                          "perturbation_size": perturbation_size})
            if self.max_changes is not None and perturbation_size > self.max_changes:
                entry["rejection_reason"] = "perturbation_budget_exceeded"
                return None, proposed, entry
            validity = validate(text, check_plausibility=True)
            entry["representation_valid"] = validity.representation_valid is True
            entry["plausible"] = validity.plausible
            if validity.representation_valid is True:
                counters["rep_valid"] += 1
            eligible = validity.representation_valid is True and (
                validity.plausible or not self.require_plausible)
            if not eligible:
                entry["rejection_reason"] = "invalid_or_implausible"
                return None, proposed, entry
            counters["valid"] += 1
            if text in cache:
                entry["rejection_reason"] = "duplicate_cached_candidate"
                return cache[text], proposed, entry
            if counters["queries"] >= self.query_budget:
                entry["rejection_reason"] = "query_budget_exhausted"
                return None, proposed, entry
            prediction = float(self.predictor.predict([text])[0])
            counters["queries"] += 1
            candidate = _Candidate(proposed.tokens, text, prediction,
                                   abs(prediction - original_prediction))
            cache[text] = candidate
            entry.update({"queried": True, "prediction": prediction,
                          "absolute_drift": candidate.drift})
            return candidate, proposed, entry

        def record(entry: dict, accepted: bool = False):
            entry["accepted"] = bool(accepted)
            if accepted:
                counters["accepted"] += 1
            trace.append(entry)

        best = self._run(original, evaluate, record)
        elapsed = time.perf_counter() - started
        applicable = self.name in {"greedy", "metropolis_mcmc"}
        return SearchResult(
            strategy=self.name, original_representation=representation,
            best_representation=best.representation, original_prediction=original_prediction,
            best_prediction=best.prediction, best_drift=best.drift,
            candidate_query_budget=self.query_budget, candidate_model_queries=counters["queries"],
            total_model_queries=counters["queries"] + 1, candidate_proposals=counters["proposals"],
            representation_valid_candidates=counters["rep_valid"], valid_candidates=counters["valid"],
            accepted_candidates=counters["accepted"],
            acceptance_rate=(counters["accepted"] / counters["valid"] if applicable and counters["valid"] else None),
            perturbation_size=count_changes(list(original.tokens), list(best.tokens)),
            elapsed_seconds=elapsed, proposal_operators=self.proposal.names, trace=trace,
        )

    @staticmethod
    def _better(left: _Candidate, right: _Candidate) -> _Candidate:
        return left if left.drift >= right.drift else right


class RandomSearch(BlackBoxSearch):
    name = "random"
    def _run(self, original, evaluate, record):
        best, current, queries = original, original, 0
        for _ in range(self.max_proposals):
            candidate, _, entry = evaluate(current)
            record(entry)
            queries += int(entry["queried"])
            if candidate is not None:
                current = candidate
                best = self._better(best, candidate)
            if queries >= self.query_budget:
                break
        return best


class GreedySearch(BlackBoxSearch):
    name = "greedy"
    def _run(self, original, evaluate, record):
        current = best = original
        queries = 0
        for _ in range(self.max_proposals):
            candidate, _, entry = evaluate(current)
            queries += int(entry["queried"])
            accepted = candidate is not None and candidate.drift > current.drift
            if accepted:
                current = candidate
                best = self._better(best, candidate)
            record(entry, accepted)
            if queries >= self.query_budget: break
        return best


class BeamSearch(BlackBoxSearch):
    name = "beam"
    def __init__(self, *args, beam_width: int = 4, branching_factor: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if beam_width < 1 or branching_factor < 1:
            raise ValueError("beam_width and branching_factor must be positive")
        self.beam_width, self.branching_factor = beam_width, branching_factor

    def _run(self, original, evaluate, record):
        beam, best, queries = [original], original, 0
        while queries < self.query_budget:
            generation = []
            for parent in beam:
                for _ in range(self.branching_factor):
                    candidate, _, entry = evaluate(parent)
                    queries += int(entry["queried"])
                    record(entry)
                    if candidate is not None:
                        generation.append(candidate)
                        best = self._better(best, candidate)
                    if queries >= self.query_budget or entry["proposal_index"] >= self.max_proposals:
                        break
                if queries >= self.query_budget or entry["proposal_index"] >= self.max_proposals: break
            if not generation:
                if entry["proposal_index"] >= self.max_proposals: break
                continue
            unique = {candidate.representation: candidate for candidate in beam + generation}
            beam = sorted(unique.values(), key=lambda item: item.drift, reverse=True)[:self.beam_width]
        return best


class MetropolisSearch(BlackBoxSearch):
    """Metropolis-style heuristic; not claimed as formal MH without proposal correction."""
    name = "metropolis_mcmc"
    def __init__(self, *args, temperature: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if temperature <= 0: raise ValueError("temperature must be positive")
        self.temperature = temperature

    def _run(self, original, evaluate, record):
        current = best = original
        queries = 0
        for _ in range(self.max_proposals):
            candidate, _, entry = evaluate(current)
            queries += int(entry["queried"])
            accepted = False
            if candidate is not None:
                delta = candidate.drift - current.drift
                probability = min(1.0, math.exp(delta / self.temperature))
                entry["acceptance_probability"] = probability
                accepted = bool(self.rng.random() < probability)
                if accepted: current = candidate
                best = self._better(best, candidate)
            record(entry, accepted)
            if queries >= self.query_budget: break
        return best


class EvolutionarySearch(BlackBoxSearch):
    name = "evolutionary"
    def __init__(self, *args, population_size: int = 8, elite_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if population_size < 1 or not 1 <= elite_size <= population_size:
            raise ValueError("invalid population or elite size")
        self.population_size, self.elite_size = population_size, elite_size

    def _run(self, original, evaluate, record):
        population, best, queries = [original], original, 0
        while queries < self.query_budget:
            offspring = []
            for _ in range(self.population_size):
                parent = population[int(self.rng.integers(0, len(population)))]
                candidate, _, entry = evaluate(parent)
                queries += int(entry["queried"])
                record(entry, False)
                if candidate is not None:
                    offspring.append(candidate)
                    best = self._better(best, candidate)
                if queries >= self.query_budget or entry["proposal_index"] >= self.max_proposals: break
            if not offspring:
                if entry["proposal_index"] >= self.max_proposals: break
                continue
            unique = {candidate.representation: candidate for candidate in population + offspring}
            elites = sorted(unique.values(), key=lambda item: item.drift, reverse=True)[:self.elite_size]
            population = elites
        return best
