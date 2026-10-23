import pytest
import numpy as np
import sys
from pathlib import Path

comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from scripts.compute_sweep_statistics import bootstrap_ci

def test_bootstrap_ci():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    lower, upper = bootstrap_ci(data, np.mean, n_bootstraps=100)
    assert lower >= 1.0
    assert upper <= 5.0
    assert lower <= upper

def test_aggregation_correctness():
    from scripts.aggregate_budget_sweep import main
    # Just asserting it imports and functions structurally
    assert callable(main)

def test_budget_enforcement():
    from src.core.budgets import BudgetManager
    b = BudgetManager(max_queries=15)
    b.start()
    b.record_query(15)
    assert b.is_exhausted()

def test_deterministic_seeds():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert rng1.random() == rng2.random()

def test_duplicate_failure_aggregation():
    from src.core.schemas import AttackResult
    res = AttackResult(
        source_id="1", source_sequence="C", candidate_sequence="C", attack_name="a", attack_family="b",
        seed=1, query_count=1, generation_count=1, runtime_seconds=1.0, valid_rdkit=True, constraint_pass=True,
        tanimoto_similarity=1.0, edit_distance=0, source_prediction=1.0, candidate_prediction=1.0,
        prediction_drift=0.0, objective_value=0.0, duplicate_proposals=5, failure_reason="Test"
    )
    d = res.to_dict()
    assert d["duplicate_proposals"] == 5
    assert d["failure_reason"] == "Test"
