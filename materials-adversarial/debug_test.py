import pytest
import numpy as np
from src.materials_adv.attacks.search.proposals import Proposal
from src.materials_adv.attacks.search.strategies import GreedySearch, _Candidate
from tests.test_phase8 import DummyPredictor

rng = np.random.default_rng(42)
vocab = ["C", "N", "O"]

class MockProposal:
    @property
    def names(self): return ("mock",)
    def enumerate_proposals(self, tokens):
        return [
            Proposal(tuple("CONC"), "mock", None),
            Proposal(tuple("NONC"), "mock", None)
        ]

proposal = MockProposal()
predictor = DummyPredictor("O")

search = GreedySearch(predictor, proposal, rng, query_budget=10, max_changes=3)

res = search.search("CCNC")
print("Best:", res.best_representation)
for t in res.trace:
    print(t)
