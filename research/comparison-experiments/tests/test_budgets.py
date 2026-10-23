import pytest
from src.core.budgets import BudgetManager

def test_budget_manager_queries():
    budget = BudgetManager(max_queries=5)
    budget.start()
    assert not budget.is_exhausted()
    budget.record_query(3)
    assert not budget.is_exhausted()
    budget.record_query(2)
    assert budget.is_exhausted()

def test_budget_manager_generations():
    budget = BudgetManager(max_queries=10, max_generations=5)
    budget.start()
    assert not budget.is_exhausted()
    budget.record_generation(4)
    assert not budget.is_exhausted()
    budget.record_generation(2)
    assert budget.is_exhausted()

def test_budget_manager_runtime():
    import time
    budget = BudgetManager(max_runtime_seconds=0.1)
    budget.start()
    assert not budget.is_exhausted()
    time.sleep(0.15)
    assert budget.is_exhausted()
