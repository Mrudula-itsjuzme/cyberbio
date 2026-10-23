import sys
from pathlib import Path
import math

# Adjust path
comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

repo_root = comp_exp_root.parent / "materials-adversarial"
if str(repo_root) not in sys.path and repo_root.exists():
    sys.path.insert(0, str(repo_root / "src"))

from src.adapters.canonical_model import CanonicalModelAdapter


def test_canonical_model_prediction():
    adapter = CanonicalModelAdapter()
    
    # known valid source
    source_seq = "[*]CC([*])c1ccccc1"
    
    pred = adapter.predict(source_seq)
    
    assert isinstance(pred, float)
    assert math.isfinite(pred)
    assert pred > 0.0, "Bandgap should be positive"

def test_model_batch_query_budget(monkeypatch):
    import numpy as np
    from src.core.budgets import BudgetManager
    # We will mock the predictor to avoid loading big checkpoint
    model_adapter = CanonicalModelAdapter()
    class DummyPredictor:
        def predict(self, seqs):
            return np.ones(len(seqs))
    model_adapter.predictor = DummyPredictor()
    
    budget = BudgetManager(max_queries=10)
    # mock predict_batch which doesn't exist yet! Ah wait, I added predict_batch in mcmc proxy, but does canonical model have predict_batch?
    
