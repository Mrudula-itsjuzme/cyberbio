import pytest
import numpy as np
import sys
from pathlib import Path

comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from src.core.budgets import BudgetManager
from src.attackers.random_search import RandomSearchAttacker
from src.attackers.evolutionary import EvolutionaryAttacker
from src.attackers.llm_guided import LLMGuidedAttacker
from src.attackers.llm_providers import MockLLMProvider
from src.adapters.canonical_mcmc import CanonicalMCMCAdapter
from src.adapters.canonical_validator import CanonicalValidatorAdapter

class CountingModel:
    def predict(self, seq):
        return 1.0
    def predict_batch(self, seqs):
        return [1.0 for _ in seqs]

def test_source_query_parity():
    model = CountingModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    rng = np.random.default_rng(42)
    
    attackers = [
        RandomSearchAttacker(),
        EvolutionaryAttacker(),
        CanonicalMCMCAdapter(),
        LLMGuidedAttacker(provider=MockLLMProvider(mode="empty"))
    ]
    
    for attacker in attackers:
        # Give a budget of exactly 1. It should only evaluate the source and then exhaust!
        budget = BudgetManager(max_queries=1)
        res = attacker.attack("1", "C", model, validator, budget, rng)
        assert res.query_count == 1
        assert budget.queries_used == 1
