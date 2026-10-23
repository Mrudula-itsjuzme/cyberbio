import pytest
import numpy as np
import sys
import json
from pathlib import Path

comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from src.core.budgets import BudgetManager
from src.attackers.llm_guided import LLMGuidedAttacker
from src.attackers.llm_providers import MockLLMProvider
from src.adapters.canonical_validator import CanonicalValidatorAdapter

class DummyModel:
    def predict(self, seq):
        return 1.5 if "1" in seq else 1.0

def test_llm_guided_malformed_json():
    attacker = LLMGuidedAttacker(provider=MockLLMProvider(mode="malformed"))
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    budget = BudgetManager(max_queries=5)
    rng = np.random.default_rng(42)
    
    res = attacker.attack("1", "C", model, validator, budget, rng)
    assert res.failure_reason == "malformed_json"
    assert res.query_count == 1 # Only the source query

def test_llm_guided_empty_response():
    attacker = LLMGuidedAttacker(provider=MockLLMProvider(mode="empty"))
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    budget = BudgetManager(max_queries=5)
    rng = np.random.default_rng(42)
    
    res = attacker.attack("1", "C", model, validator, budget, rng)
    assert res.failure_reason == "empty_response"
    assert res.query_count == 1

def test_llm_guided_valid_response():
    attacker = LLMGuidedAttacker(provider=MockLLMProvider(mode="valid"), max_proposals_per_call=2)
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    budget = BudgetManager(max_queries=5)
    rng = np.random.default_rng(42)
    
    res = attacker.attack("1", "C", model, validator, budget, rng)
    assert res.query_count > 1
    assert res.llm_calls > 0
    assert res.llm_input_tokens > 0
    assert res.llm_output_tokens > 0

def test_prompt_modes():
    provider = MockLLMProvider(mode="valid")
    for mode in ["blind", "objective-aware", "iterative", "operator-constrained"]:
        attacker = LLMGuidedAttacker(provider=provider, prompt_mode=mode)
        prompt = attacker._build_prompt("C", history=[{"cand": "CC", "valid": True, "drift": 0.5}])
        assert mode in prompt or "Objective" in prompt or "operator" in prompt or "diverse" in prompt

def test_duplicate_filtering():
    # If the provider always returns the same mock candidates
    class RepeatingMock(MockLLMProvider):
        def generate(self, prompt, max_proposals, seed=None):
            return {
                "content": json.dumps({"proposals": [{"sequence": "CC", "edit_description": "d", "rationale": "r"}]}),
                "calls": 1
            }
            
    attacker = LLMGuidedAttacker(provider=RepeatingMock())
    model = DummyModel()
    validator = CanonicalValidatorAdapter(check_plausibility=False)
    budget = BudgetManager(max_queries=5)
    rng = np.random.default_rng(42)
    
    res = attacker.attack("1", "C", model, validator, budget, rng)
    assert res.duplicate_proposals > 0
    assert res.query_count == 2 # Source + the first unique candidate, then loop hits duplicates

def test_llm_guided_rate_limit():
    class RateLimitMock(MockLLMProvider):
        def generate(self, prompt, max_proposals, seed=None):
            return {"error": "Rate limit exceeded", "calls": 1, "content": ""}
            
    attacker = LLMGuidedAttacker(provider=RateLimitMock())
    res = attacker.attack("1", "C", DummyModel(), CanonicalValidatorAdapter(check_plausibility=False), BudgetManager(max_queries=5), np.random.default_rng(42))
    assert res.failure_reason == "provider_error" and "Rate limit" in res.metadata.get("provider_error_msg", "")

def test_llm_guided_token_accounting():
    attacker = LLMGuidedAttacker(provider=MockLLMProvider(mode="valid"), max_proposals_per_call=2)
    res = attacker.attack("1", "C", DummyModel(), CanonicalValidatorAdapter(check_plausibility=False), BudgetManager(max_queries=5), np.random.default_rng(42))
    assert res.llm_input_tokens > 0
    assert res.llm_output_tokens > 0
    assert res.llm_calls > 0

def test_llm_guided_history_metadata():
    attacker = LLMGuidedAttacker(provider=MockLLMProvider(mode="valid"), max_proposals_per_call=2)
    res = attacker.attack("1", "C", DummyModel(), CanonicalValidatorAdapter(check_plausibility=False), BudgetManager(max_queries=5), np.random.default_rng(42))
    assert "history" in res.metadata
    assert len(res.metadata["history"]) > 0
    assert "rationale" in res.metadata

def test_llm_guided_budget_exhausted():
    # Provider always gives duplicates
    class DupeMock(MockLLMProvider):
        def generate(self, prompt, max_proposals, seed=None):
            return {"content": json.dumps({"proposals": [{"sequence": "C", "edit_description": "d", "rationale": "r"}]}), "calls": 1}
    attacker = LLMGuidedAttacker(provider=DupeMock())
    res = attacker.attack("1", "C", DummyModel(), CanonicalValidatorAdapter(check_plausibility=False), BudgetManager(max_queries=5), np.random.default_rng(42))
    assert res.failure_reason == "budget_exhausted" or res.failure_reason == "no_improvement"

def test_llm_guided_timeout():
    class TimeoutMock(MockLLMProvider):
        def generate(self, prompt, max_proposals, seed=None):
            return {"error": "timeout", "calls": 1, "content": ""}
            
    attacker = LLMGuidedAttacker(provider=TimeoutMock())
    res = attacker.attack("1", "C", DummyModel(), CanonicalValidatorAdapter(check_plausibility=False), BudgetManager(max_queries=5), np.random.default_rng(42))
    assert res.failure_reason == "provider_error"
    assert "timeout" in res.metadata.get("provider_error_msg", "")

def test_llm_guided_proposal_rates():
    attacker = LLMGuidedAttacker(provider=MockLLMProvider(mode="valid"), max_proposals_per_call=2)
    res = attacker.attack("1", "C", DummyModel(), CanonicalValidatorAdapter(check_plausibility=False), BudgetManager(max_queries=5), np.random.default_rng(42))
    assert res.total_proposals > 0
    assert res.parsed_proposals >= 0
    assert res.unique_proposals >= 0

def test_llm_guided_failure_enum():
    class JsonErrorMock(MockLLMProvider):
        def generate(self, prompt, max_proposals, seed=None):
            return {"content": "not json", "calls": 1}
    
    attacker = LLMGuidedAttacker(provider=JsonErrorMock())
    res = attacker.attack("1", "C", DummyModel(), CanonicalValidatorAdapter(check_plausibility=False), BudgetManager(max_queries=5), np.random.default_rng(42))
    assert res.failure_reason in ["malformed_json", "budget_exhausted"]
