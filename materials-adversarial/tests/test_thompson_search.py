import pytest
import numpy as np
from materials_adv.attacks.search.strategies import ThompsonSamplingSearch, UniformSamplingSearch
from materials_adv.attacks.search.proposals import Proposal
from materials_adv.attacks.base import AttackOutcome
from materials_adv.attacks.search.strategies import _Candidate
from materials_adv.validation.pipeline import validate
import materials_adv.attacks.search.strategies as strats

class MockProposal:
    def __init__(self, name, returns_valid=True, mock_drift=0.0):
        self._name = name
        self.returns_valid = returns_valid
        self.mock_drift = mock_drift
        
    @property
    def names(self):
        return (self._name,)
        
    def propose(self, tokens):
        if self.returns_valid:
            new_tokens = tuple(list(tokens) + [f"[{self._name}]"])
            outcome = AttackOutcome(original_tokens=tuple(tokens), adversarial_tokens=new_tokens, attack_type=self._name)
            return Proposal(tokens=new_tokens, operator=self._name, outcome=outcome)
        return None
        
    def enumerate_proposals(self, tokens):
        return []

class MockPredictor:
    def __init__(self, proposals_dict):
        self.proposals_dict = proposals_dict
        self.call_count = 0
    def predict(self, representations):
        self.call_count += len(representations)
        out = []
        for rep in representations:
            drift = 1.0
            for name, prop in self.proposals_dict.items():
                if f"[{name}]" in rep:
                    drift = prop.mock_drift
            out.append(drift)
        return np.array(out)
    @property
    def target_units(self):
        return "mock"

class DummyValidity:
    def __init__(self, valid=True):
        self.representation_valid = valid
        self.plausible = True

@pytest.fixture(autouse=True)
def mock_validate(monkeypatch):
    # Mock validate so it doesn't try to parse 'C[good]' with RDKit
    def fake_validate(text, check_plausibility=True):
        return DummyValidity(valid=True)
    monkeypatch.setattr(strats, "validate", fake_validate)

# Test A: Posterior update correctness
def test_posterior_updates():
    good_arm = MockProposal("good", returns_valid=True, mock_drift=100.0)
    bad_arm = MockProposal("bad", returns_valid=True, mock_drift=1.0)
    predictor = MockPredictor({"good": good_arm, "bad": bad_arm})
    rng = np.random.default_rng(42) # Deterministic
    
    search = ThompsonSamplingSearch(
        predictor=predictor,
        proposals=[good_arm, bad_arm],
        rng=rng,
        query_budget=2,
        require_plausible=False,
        reward_mode="binary",
        threshold=50.0
    )
    
    result = search.search("C")
    
    for entry in result.trace:
        a_before = entry["alpha_before"]
        b_before = entry["beta_before"]
        reward = entry["reward"]
        a_after = entry["alpha_after"]
        b_after = entry["beta_after"]
        
        assert a_after == a_before + reward
        assert b_after == b_before + (1.0 - reward)

# Test B: Better arm receives higher posterior expectation
def test_expected_value_ordering():
    good_arm = MockProposal("good", returns_valid=True, mock_drift=100.0)
    bad_arm = MockProposal("bad", returns_valid=True, mock_drift=1.0)
    predictor = MockPredictor({"good": good_arm, "bad": bad_arm})
    rng = np.random.default_rng(42)
    
    search = ThompsonSamplingSearch(
        predictor=predictor,
        proposals=[good_arm, bad_arm],
        rng=rng,
        query_budget=50,
        require_plausible=False,
        reward_mode="binary",
        threshold=50.0
    )
    
    result = search.search("C")
    
    alpha_0, beta_0 = 1.0, 1.0
    alpha_1, beta_1 = 1.0, 1.0
    
    for entry in result.trace:
        if entry["chosen_arm"] == 0:
            alpha_0 = entry["alpha_after"]
            beta_0 = entry["beta_after"]
        else:
            alpha_1 = entry["alpha_after"]
            beta_1 = entry["beta_after"]
            
    e_good = alpha_0 / (alpha_0 + beta_0)
    e_bad = alpha_1 / (alpha_1 + beta_1)
    
    assert e_good > e_bad

# Test C: Invalid arm penalty
def test_invalid_arm_penalty():
    invalid_arm = MockProposal("invalid", returns_valid=False)
    predictor = MockPredictor({"invalid": invalid_arm})
    rng = np.random.default_rng(42)
    
    search = ThompsonSamplingSearch(
        predictor=predictor,
        proposals=[invalid_arm],
        rng=rng,
        query_budget=5,
        require_plausible=False,
        reward_mode="validity-aware binary"
    )
    
    result = search.search("C")
    
    assert result.candidate_model_queries == 0
    assert result.candidate_proposals == 250
    
    for entry in result.trace:
        assert entry["reward"] == 0.0
        assert entry["target_query_count"] == 0
        
# Test D: Budget enforcement

def test_budget_enforcement():
    # To test exactly hitting the budget, we need it to propose unique candidates
    class UniqueMockProposal(MockProposal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.count = 0
        def propose(self, tokens):
            self.count += 1
            new_tokens = tuple(list(tokens) + [f"[{self._name}{self.count}]"])
            outcome = AttackOutcome(original_tokens=tuple(tokens), adversarial_tokens=new_tokens, attack_type=self._name)
            return Proposal(tokens=new_tokens, operator=self._name, outcome=outcome)
            
    unique_arm = UniqueMockProposal("good", returns_valid=True, mock_drift=100.0)
    predictor = MockPredictor({"good": unique_arm})
    rng = np.random.default_rng(42)
    budget = 7
    search = ThompsonSamplingSearch(
        predictor=predictor,
        proposals=[unique_arm],
        rng=rng,
        query_budget=budget,
        proposal_budget=50,
        require_plausible=False
    )
    result = search.search("C")
    assert result.candidate_model_queries == budget
    assert result.candidate_proposals >= budget
    assert result.candidate_proposals <= 50


def test_reproducibility():
    good_arm = MockProposal("good", returns_valid=True, mock_drift=100.0)
    bad_arm = MockProposal("bad", returns_valid=True, mock_drift=1.0)
    predictor = MockPredictor({"good": good_arm, "bad": bad_arm})
    
    def run_search(seed):
        rng = np.random.default_rng(seed)
        search = ThompsonSamplingSearch(
            predictor=predictor,
            proposals=[good_arm, bad_arm],
            rng=rng,
            query_budget=15,
            require_plausible=False,
            reward_mode="binary",
            threshold=50.0
        )
        return search.search("C")
        
    res1 = run_search(123)
    res2 = run_search(123)
    res3 = run_search(999)
    
    choices1 = [e["chosen_arm"] for e in res1.trace]
    choices2 = [e["chosen_arm"] for e in res2.trace]
    choices3 = [e["chosen_arm"] for e in res3.trace]
    
    assert choices1 == choices2
    assert choices1 != choices3


def test_duplicate_arm_penalty():
    # A mock proposal that always generates the exact same sequence
    # meaning all proposals after the first will be duplicates.
    good_arm = MockProposal("good", returns_valid=True, mock_drift=100.0)
    predictor = MockPredictor({"good": good_arm})
    rng = np.random.default_rng(42)
    
    search = ThompsonSamplingSearch(
        predictor=predictor,
        proposals=[good_arm],
        rng=rng,
        query_budget=5,
        proposal_budget=10,
        require_plausible=False,
        reward_mode="binary",
        threshold=50.0
    )
    result = search.search("C")
    
    # Trace length should be 10 (since proposal budget is 10 and it loops proposing same thing)
    assert len(result.trace) == 10
    
    # First proposal is fresh, posterior should be updated
    assert result.trace[0]["posterior_updated"] is True
    assert result.trace[0]["duplicate"] is False
    assert result.trace[0]["reward"] == 1.0
    assert result.trace[0]["alpha_after"] == 2.0
    assert result.trace[0]["beta_after"] == 1.0
    
    # 2nd through 10th proposals are duplicates
    for i in range(2, 10):
        assert result.trace[i]["posterior_updated"] is False
        assert result.trace[i]["duplicate"] is True
        assert result.trace[i]["reward"] == 0.0
        assert result.trace[i]["alpha_after"] == 3.0
        assert result.trace[i]["beta_after"] == 1.0
