import pytest
from src.core.budgets import BudgetManager
from src.core.schemas import AttackResult
from src.adapters.canonical_validator import CanonicalValidatorAdapter

def test_budget_manager_edge_case_runtime():
    b = BudgetManager(max_runtime_seconds=0.01)
    b.start()
    import time
    time.sleep(0.02)
    assert b.is_exhausted()

def test_schema_metadata_defaults():
    res = AttackResult(
        source_id="1", source_sequence="C", candidate_sequence="C", attack_name="a", attack_family="b",
        seed=1, query_count=1, generation_count=1, runtime_seconds=1.0, valid_rdkit=True, constraint_pass=True,
        tanimoto_similarity=1.0, edit_distance=0, source_prediction=1.0, candidate_prediction=1.0,
        prediction_drift=0.0, objective_value=0.0
    )
    assert res.metadata == {}

def test_validator_plausibility_flag():
    val = CanonicalValidatorAdapter(check_plausibility=False)
    # RDKit validation should still work but plausible defaults or skips
    assert val.is_valid_plausible("[*]CC([*])C")

def test_validator_invalid_smiles_edge_cases():
    val = CanonicalValidatorAdapter(check_plausibility=True)
    assert not val.is_valid_rdkit("C(C))")
    assert not val.is_valid_rdkit("12345")
