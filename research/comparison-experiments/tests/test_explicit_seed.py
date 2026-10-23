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

class DummyModel:
    def predict(self, seq): return 1.0
    def predict_batch(self, seqs): return [1.0] * len(seqs)

def test_explicit_seed_propagation():
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    
    attackers = [
        RandomSearchAttacker(),
        EvolutionaryAttacker(),
        CanonicalMCMCAdapter(),
        LLMGuidedAttacker(provider=MockLLMProvider(mode="empty"))
    ]
    
    test_seed = 999
    rng = np.random.default_rng(test_seed)
    
    for attacker in attackers:
        budget = BudgetManager(max_queries=1)
        res = attacker.attack("1", "C", model, validator, budget, rng, seed=test_seed)
        assert res.seed == test_seed, f"{attacker.name} did not record the explicit seed!"

