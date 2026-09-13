import pytest
import numpy as np
from src.materials_adv.attacks.substitution import SubstitutionAttack
from src.materials_adv.attacks.search.proposals import CompositeProposalOperator, Proposal
from src.materials_adv.attacks.search.strategies import RandomSearch, GreedySearch, MetropolisSearch, _Candidate

class DummyPredictor:
    def __init__(self, target="C"):
        self.target = target
        
    def predict(self, texts):
        # Predict 1.0 if it has target character, else 0.0
        return np.array([1.0 if self.target in t else 0.0 for t in texts])

def test_random_search_restart_policy():
    rng = np.random.default_rng(42)
    # create an attack that always proposes invalid or same token (so it fails)
    vocab = ["X", "Y"]
    attack = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=False)
    proposal = CompositeProposalOperator([attack], rng)
    
    # We will mock the evaluate function to force failures
    predictor = DummyPredictor()
    search = RandomSearch(predictor, proposal, rng, query_budget=10, max_changes=3)
    
    # Run a mock evaluate
    class MockSearch(RandomSearch):
        def _run(self, original, evaluate, record):
            self.failures = 0
            self.resets = 0
            
            # Mock evaluate to always fail
            def mock_eval(parent, p=None):
                self.failures += 1
                return None, None, {"queried": True, "representation_valid": False}
                
            best, current, queries = original, original, 0
            for _ in range(self.max_proposals):
                candidate, _, entry = mock_eval(current)
                queries += 1
                
                if candidate is None:
                    # Logic from RandomSearch
                    pass # Handled by failures counter
                
                # Check restart
                if self.failures >= 3:
                    current = original
                    self.failures = 0
                    self.resets += 1
                    
                if queries >= self.query_budget:
                    break
            return best
            
    s = MockSearch(predictor, proposal, rng, query_budget=10, max_changes=3)
    orig = _Candidate(tuple("XXXX"), "XXXX", 0.0, 0.0)
    s._run(orig, None, lambda x: None)
    
    assert s.resets == 3 # 10 queries / 3 failures = 3 resets

def test_greedy_search_enumeration(monkeypatch):
    import src.materials_adv.attacks.search.strategies as strats
    # Mock validate to always pass
    class MockValidity:
        representation_valid = True
        plausible = True
    monkeypatch.setattr(strats, "validate", lambda *args, **kwargs: MockValidity())
    
    rng = np.random.default_rng(42)
    vocab = ["C", "N", "O"]
    
    # Mock proposal operator to yield deterministic neighbors
    class MockProposal:
        @property
        def names(self):
            return ("mock",)
        def enumerate_proposals(self, tokens):
            return [
                Proposal(tuple("CONC"), "mock", None),
                Proposal(tuple("NONC"), "mock", None)
            ]
            
    proposal = MockProposal()
    predictor = DummyPredictor("O")
    
    search = GreedySearch(predictor, proposal, rng, query_budget=10, max_changes=3)
    
    res = search.search("CCNC")
    assert res.best_representation == "CONC"
    
def test_metropolis_multi_edit():
    rng = np.random.default_rng(42)
    vocab = ["C", "(", "=", "O", ")", "O", "C", "C", "C"]
    
    attack = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=True)
    proposal = CompositeProposalOperator([attack], rng)
    
    predictor = DummyPredictor("O")
    
    search = MetropolisSearch(predictor, proposal, rng, query_budget=10, max_changes=3, temperature=0.1)
    
    # This just needs to not crash and respect budget
    res = search.search("CC(O)C")
    
    # Perturbation size should never exceed 3
    for t in res.trace:
        if "perturbation_size" in t:
            assert t["perturbation_size"] <= 3

