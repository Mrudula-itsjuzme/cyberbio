from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from probabilistic_search_lab.accounting import CountedTargetModel, QueryLedger
from probabilistic_search_lab.contracts import MethodStatus, SearchMetrics, SearchResult
from probabilistic_search_lab.metrics import summarize_effects


@dataclass(slots=True)
class Posterior:
    alpha: float = 1.0
    beta: float = 1.0
    reward_count: int = 0
    reward_mean: float = 0.0
    reward_m2: float = 0.0

    def update_binary(self, success: bool) -> None:
        self.alpha += float(success)
        self.beta += float(not success)

    def update_continuous(self, reward: float) -> None:
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_m2 += delta * (reward - self.reward_mean)

    def sample_continuous(self, rng: np.random.Generator) -> float:
        if self.reward_count < 2:
            return float(rng.normal(self.reward_mean, 1.0))
        variance = max(self.reward_m2 / (self.reward_count - 1), 1e-12)
        return float(rng.normal(self.reward_mean, np.sqrt(variance / self.reward_count)))


class ThompsonSearch:
    """Non-contextual family selection with explicit budget/validity accounting."""

    def __init__(self, families, reward_mode: str = "binary", success_threshold: float = 0.0):
        if reward_mode not in {"binary", "continuous", "validity_aware"}:
            raise ValueError(f"unknown reward mode: {reward_mode}")
        if not families:
            raise ValueError("at least one proposal family is required")
        self.families = list(families)
        self.reward_mode = reward_mode
        self.success_threshold = success_threshold

    def run(self, *, source_id, source, model, validator, budget: int, seed: int) -> SearchResult:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        ledger = QueryLedger(budget=budget)
        target = CountedTargetModel(model, ledger)
        posteriors = {family.name: Posterior() for family in self.families}
        trajectory = []
        effects: list[float] = []
        valid_candidates: set[str] = set()
        invalid_count = 0
        best_candidate = None

        if budget < 1:
            raise ValueError("budget must include at least the clean-source query")
        source_prediction = target.predict(source)

        max_proposals = max(100, budget * 100)
        stopped_for_proposals = False
        while ledger.remaining_target > 0:
            if ledger.proposal_count >= max_proposals:
                stopped_for_proposals = True
                break
            samples = {}
            for family in self.families:
                posterior = posteriors[family.name]
                if self.reward_mode == "continuous":
                    samples[family.name] = posterior.sample_continuous(rng)
                else:
                    samples[family.name] = float(rng.beta(posterior.alpha, posterior.beta))
            selected = max(self.families, key=lambda family: samples[family.name])
            proposals = list(selected.propose(source, source, rng))
            ledger.proposal_count += len(proposals)
            if not proposals:
                posteriors[selected.name].update_binary(False)
                trajectory.append({"family": selected.name, "outcome": "no_proposal"})
                continue

            proposal = proposals[int(rng.integers(len(proposals)))]
            ledger.validity_checks += 1
            validation = validator.validate(source, proposal.candidate)
            reward = 0.0
            effect = None
            if validation.eligible:
                valid_candidates.add(proposal.candidate)
                prediction = target.predict(proposal.candidate)
                effect = abs(prediction - source_prediction)
                effects.append(effect)
                reward = effect - proposal.edit_cost
                if best_candidate is None or effect >= max(effects[:-1], default=-1.0):
                    best_candidate = proposal.candidate
            else:
                invalid_count += 1

            posterior = posteriors[selected.name]
            success = validation.eligible and effect is not None and effect > self.success_threshold
            if self.reward_mode == "continuous":
                posterior.update_continuous(reward)
            elif self.reward_mode == "validity_aware":
                posterior.update_binary(success and validation.eligible)
            else:
                posterior.update_binary(success)
            trajectory.append({
                "family": selected.name,
                "sample": samples[selected.name],
                "valid": validation.eligible,
                "effect": effect,
                "reward": reward,
                "alpha": posterior.alpha,
                "beta": posterior.beta,
            })

        summary = summarize_effects(effects)
        valid_count = len(effects)
        metrics = SearchMetrics(
            proposal_count=ledger.proposal_count,
            validity_checks=ledger.validity_checks,
            valid_count=valid_count,
            invalid_count=invalid_count,
            target_model_queries=ledger.target_model_queries,
            runtime_seconds=time.perf_counter() - started,
            mean_effect=summary["mean"],
            median_effect=summary["median"],
            p90_effect=summary["p90"],
            best_effect=summary["best"],
            diversity_score=(len(valid_candidates) / valid_count) if valid_count else None,
        )
        return SearchResult(
            method=f"thompson_{self.reward_mode}",
            status=MethodStatus.FAILED if stopped_for_proposals else MethodStatus.WORKS,
            seed=seed, source_id=str(source_id), query_budget=budget,
            metrics=metrics, best_candidate=best_candidate,
            failure_status="PROPOSAL_LIMIT_WITH_UNUSED_QUERY_BUDGET" if stopped_for_proposals else None,
            trajectories={"posterior": trajectory},
            metadata={"success_threshold": self.success_threshold},
        )
