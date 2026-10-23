import pytest
import numpy as np
import sys
from pathlib import Path

comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from src.core.budgets import BudgetManager

def test_model_batch_query_budget():
    budget = BudgetManager(max_queries=10)
    budget.start()
    
    # query batch of 5
    budget.record_query(5)
    assert budget.queries_used == 5
    assert not budget.is_exhausted()
    
    # query batch of 6 -> exceeds 10
    budget.record_query(6)
    assert budget.queries_used == 11
    assert budget.is_exhausted()

def test_budget_start_multiple_times():
    budget = BudgetManager(max_queries=10)
    budget.start()
    budget.record_query(5)
    budget.start() # Should reset or no-op? Actually usually we just use it once. Let's see if start resets.
    
