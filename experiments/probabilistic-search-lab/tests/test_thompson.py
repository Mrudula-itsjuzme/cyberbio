from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "thompson")]

from probabilistic_search_lab.interfaces import Proposal, Validation
from thompson.search import ThompsonSearch


class Model:
    def predict(self, sequence: str) -> float:
        return float(len(sequence))


class Validator:
    def validate(self, source: str, candidate: str) -> Validation:
        valid = candidate.isalpha()
        return Validation(valid, valid, () if valid else ("non_alpha",))


@dataclass
class Family:
    name: str
    suffix: str

    def propose(self, source: str, current: str, rng: np.random.Generator):
        return [Proposal(current + self.suffix, self.name, edit_cost=0.0)]


def test_thompson_is_reproducible_and_budget_exact():
    kwargs = dict(
        source_id="s1", source="CC", model=Model(), validator=Validator(),
        budget=7, seed=11,
    )
    first = ThompsonSearch([Family("a", "A"), Family("b", "BB")]).run(**kwargs)
    second = ThompsonSearch([Family("a", "A"), Family("b", "BB")]).run(**kwargs)
    assert first.metrics.target_model_queries == 7
    assert first.metrics.valid_count == 6
    assert first.metrics.invalid_count == 0
    assert first.metrics.best_effect == 2.0
    assert first.trajectories == second.trajectories


def test_invalid_proposals_do_not_consume_target_queries():
    result = ThompsonSearch([Family("invalid", "1"), Family("valid", "A")]).run(
        source_id="s1", source="CC", model=Model(), validator=Validator(), budget=5, seed=4
    )
    assert result.metrics.target_model_queries == 5
    assert result.metrics.invalid_count > 0
    assert result.metrics.validity_checks == result.metrics.valid_count + result.metrics.invalid_count


def test_continuous_reward_records_missing_safe_metrics():
    result = ThompsonSearch([Family("a", "A")], reward_mode="continuous").run(
        source_id="s1", source="CC", model=Model(), validator=Validator(), budget=2, seed=1
    )
    assert result.metrics.mean_effect == 1.0
    assert result.trajectories["posterior"][0]["reward"] == 1.0
