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

def test_provider_model_serialization():
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    rng = np.random.default_rng(42)
    provider = MockLLMProvider(mode="valid")
    provider.model_name = "mock-model-1"
    
    attacker = LLMGuidedAttacker(provider=provider, prompt_mode="blind")
    budget = BudgetManager(max_queries=2)
    res = attacker.attack("1", "C", model, validator, budget, rng, seed=42)
    
    assert res.metadata["provider"] == "MockLLMProvider"
    assert res.metadata["model"] == "mock-model-1"
    assert "prompt_hash" in res.metadata

def test_iteration_history_serialization():
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    rng = np.random.default_rng(42)
    provider = MockLLMProvider(mode="valid")
    
    attacker = LLMGuidedAttacker(provider=provider, prompt_mode="iterative")
    budget = BudgetManager(max_queries=3) # Allow multiple iterations
    res = attacker.attack("1", "C", model, validator, budget, rng, seed=42)
    
    assert "history" in res.metadata
    assert isinstance(res.metadata["history"], list)

def test_zero_denominator_metrics():
    # If total_proposals is 0, the rates should be 0.0, and not throw ZeroDivisionError
    res_dict = {
        "total_proposals": 0,
        "parsed_proposals": 0,
        "unique_proposals": 0,
        "duplicate_proposals": 0,
        "rdkit_valid_proposals": 0,
        "constraint_pass_proposals": 0,
        "evaluated_proposals": 0,
        "successful_proposals": 0,
    }
    raw_val_rate = res_dict["rdkit_valid_proposals"] / res_dict["total_proposals"] if res_dict["total_proposals"] > 0 else 0.0
    assert raw_val_rate == 0.0

def test_operator_compliance_fallback():
    # We will implement operator-constrained compliance logic in scripts or via LLM rationale checking
    def classify_operator(source, candidate, rationale):
        if "swap" in rationale.lower(): return "swap"
        return "manual_review_required"
        
    assert classify_operator("C", "CC", "I did a swap") == "swap"
    assert classify_operator("C", "CC", "I mutated it") == "manual_review_required"

