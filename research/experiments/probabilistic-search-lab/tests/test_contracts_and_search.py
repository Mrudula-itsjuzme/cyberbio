from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import EvolutionarySearch, RandomSearch
from cem import CEMSearch
from probabilistic_search_lab.accounting import CountedTargetModel, QueryBudgetExceeded, QueryLedger
from probabilistic_search_lab.canonical import CanonicalValidator
from probabilistic_search_lab.interfaces import Proposal, Validation
from smc import SMCSearch
from validity_smc import EmpiricalValidityModel


class Model:
    def predict(self, sequence): return float(len(sequence))
    def predict_batch(self, sequences): return [self.predict(s) for s in sequences]


class Validator:
    def validate(self, source, candidate):
        valid = candidate.isalpha()
        return Validation(valid, valid, () if valid else ("invalid",))


@dataclass
class Family:
    name: str
    suffix: str
    def propose(self, source, current, rng):
        return [Proposal(current + self.suffix, self.name, 0.0, (self.name,))]


def test_counted_model_is_single_budget_gateway():
    ledger = QueryLedger(2); model = CountedTargetModel(Model(), ledger)
    assert model.predict("C") == 1.0
    assert model.predict_batch(["CC"]) == [2.0]
    assert ledger.target_model_queries == 2
    try:
        model.predict("CCC")
    except QueryBudgetExceeded:
        pass
    else:
        raise AssertionError("over-budget query was not rejected")


def test_validation_missing_is_not_eligible():
    assert not Validation(None, None).eligible
    assert not Validation(True, None).eligible
    assert Validation(True, True).eligible


def test_canonical_validator_uses_source_relative_contract():
    source = "[*]CC[*]"
    validator = CanonicalValidator(min_tanimoto_similarity=0.0)
    assert validator.validate(source, source).eligible
    broken = validator.validate(source, "CC")
    assert broken.representation_valid is True
    assert broken.domain_constraint_pass is False
    assert not broken.eligible


def test_searches_respect_exact_target_budget():
    families = [Family("one", "A"), Family("two", "BB")]
    searches = [RandomSearch(families), EvolutionarySearch(families, population_size=2),
                CEMSearch(families, batch_size=2), SMCSearch(families, particles=2),
                SMCSearch(families, particles=2, validity_model=EmpiricalValidityModel())]
    for search in searches:
        result = search.run(source_id="x", source="C", model=Model(), validator=Validator(), budget=5, seed=3)
        assert result.metrics.target_model_queries == 5, result.method
        assert result.metrics.valid_count == 4, result.method
        assert result.metrics.best_effect is not None


def test_validity_model_uses_validity_labels_only():
    model = EmpiricalValidityModel(); model.update("a", True); model.update("a", False)
    assert model.probability("a") == 0.5
    assert model.snapshot()["a"] == {"alpha": 2.0, "beta": 2.0, "p_valid": 0.5}
