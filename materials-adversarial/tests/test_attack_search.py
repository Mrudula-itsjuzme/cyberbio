import numpy as np
import pytest

from materials_adv.attacks.base import AttackOutcome
from materials_adv.attacks.search import (
    BeamSearch, EvolutionarySearch, GreedySearch, MetropolisSearch, Proposal, RandomSearch,
)
from materials_adv.data.tokenizer import tokenize
from materials_adv.experiments.attack_efficiency import (
    AttackEfficiencyBenchmark, AttackEfficiencySettings,
)


class CountingPredictor:
    target_units = "eV"
    def __init__(self): self.queries = 0
    def predict(self, strings):
        self.queries += len(strings)
        return np.asarray([s.count("C") for s in strings], dtype=float)


class UniqueCarbonProposal:
    names = ("test_insert",)
    def __init__(self): self.index = 1
    def propose(self, tokens):
        self.index += 1
        original = tuple(tokens)
        candidate = tuple(tokenize("[*]" + "C" * self.index + "[*]"))
        outcome = AttackOutcome(original, candidate, "test_insert")
        return Proposal(candidate, "test_insert", outcome)


@pytest.mark.parametrize("strategy,kwargs", [
    (RandomSearch, {}), (GreedySearch, {}),
    (BeamSearch, {"beam_width": 2, "branching_factor": 2}),
    (MetropolisSearch, {"temperature": 1.0}),
    (EvolutionarySearch, {"population_size": 3, "elite_size": 2}),
])
def test_search_strategies_obey_identical_candidate_query_budget(strategy, kwargs):
    predictor = CountingPredictor()
    search = strategy(predictor, UniqueCarbonProposal(), np.random.default_rng(7),
                      query_budget=5, require_plausible=False, **kwargs)
    result = search.search("[*]C[*]")
    assert result.candidate_query_budget == 5
    assert result.candidate_model_queries == 5
    assert result.total_model_queries == 6
    assert predictor.queries == 6
    assert result.summary()["query_budget_exhausted"] is True
    assert result.best_drift == 5.0
    assert result.drift_per_model_query == pytest.approx(1.0)
    assert result.candidate_proposals == 5


def test_invalid_proposals_do_not_consume_model_queries():
    class InvalidProposal:
        names = ("invalid",)
        def propose(self, tokens):
            bad = tuple(tokenize("C("))
            return Proposal(bad, "invalid", AttackOutcome(tuple(tokens), bad, "invalid"))
    predictor = CountingPredictor()
    result = RandomSearch(predictor, InvalidProposal(), np.random.default_rng(1),
                          query_budget=3, max_proposal_multiplier=2).search("[*]C[*]")
    assert result.candidate_model_queries == 0
    assert result.total_model_queries == 1
    assert result.candidate_proposals == 6
    assert result.representation_valid_candidates == 0


def test_metropolis_records_acceptance_probability_and_rate():
    result = MetropolisSearch(CountingPredictor(), UniqueCarbonProposal(),
                              np.random.default_rng(2), query_budget=3,
                              require_plausible=False, temperature=1.0).search("[*]C[*]")
    assert result.acceptance_rate == 1.0
    assert all("acceptance_probability" in row for row in result.trace)


def test_efficiency_benchmark_persists_comparable_budget_artifacts(tmp_path):
    predictor = CountingPredictor()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "run"
    summary = AttackEfficiencyBenchmark(AttackEfficiencySettings(
        query_budget=2, max_changes=10, require_plausible=False,
        population_size=2, elite_size=1, beam_width=2, branching_factor=1,
    )).run(samples=[("sample", "[*]C[*]")], predictor=predictor,
           proposal_factory=lambda rng: UniqueCarbonProposal(), output_dir=output,
           checkpoint_path=checkpoint, metadata={"target_units": "eV"},
           config_snapshot={"test": True})
    assert summary["configured_equal_candidate_query_budget"] is True
    assert summary["all_runs_exhausted_budget"] is True
    assert len(summary["search_strategies"]) == 5
    assert (output / "proposal_trace.jsonl").is_file()
    assert (output / "query_efficiency_curves.csv").is_file()
    with pytest.raises(FileExistsError):
        AttackEfficiencyBenchmark(AttackEfficiencySettings()).run(
            samples=[], predictor=predictor, proposal_factory=lambda rng: UniqueCarbonProposal(),
            output_dir=output, checkpoint_path=checkpoint, metadata={}, config_snapshot={})
