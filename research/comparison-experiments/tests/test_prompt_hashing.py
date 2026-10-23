import pytest
import numpy as np
import sys
from pathlib import Path

comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from src.core.budgets import BudgetManager
from src.attackers.llm_guided import LLMGuidedAttacker
from src.attackers.llm_providers import MockLLMProvider
from src.adapters.canonical_validator import CanonicalValidatorAdapter

class DummyModel:
    def predict(self, seq): return 1.0
    def predict_batch(self, seqs): return [1.0] * len(seqs)

def test_prompt_hashing():
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    rng = np.random.default_rng(42)
    
    # Mode that definitely exists
    attacker = LLMGuidedAttacker(provider=MockLLMProvider(mode="empty"), prompt_mode="blind")
    budget = BudgetManager(max_queries=2)
    res = attacker.attack("1", "C", model, validator, budget, rng, seed=42)
    
    assert "prompt_hash" in res.metadata
    assert len(res.metadata["prompt_hash"]) > 10
