import pytest
import numpy as np
import sys
from pathlib import Path

comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from src.core.budgets import BudgetManager
from src.attackers.evolutionary import EvolutionaryAttacker
from src.adapters.canonical_validator import CanonicalValidatorAdapter

class DummyModel:
    def predict(self, seq):
        return 1.5

def test_evolutionary_search():
    attacker = EvolutionaryAttacker(population_size=2)
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    budget = BudgetManager(max_queries=5)
    rng = np.random.default_rng(42)
    
    # Run attack
    res = attacker.attack(
        source_id="1",
        source_sequence="[*]CC([*])c1ccccc1",
        model=model,
        validator=validator,
        budget=budget,
        rng=rng
    )
    
    assert res.attack_family == "evolutionary"
    assert res.query_count > 0
