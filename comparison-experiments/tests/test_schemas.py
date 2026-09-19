import pytest
from src.core.schemas import AttackResult

def test_schema_serialization():
    res = AttackResult(
        source_id="1",
        source_sequence="CC",
        candidate_sequence="CCC",
        attack_name="test",
        attack_family="test_fam",
        seed=1,
        query_count=2,
        generation_count=5,
        runtime_seconds=1.0,
        valid_rdkit=True,
        constraint_pass=False,
        tanimoto_similarity=0.8,
        edit_distance=1,
        source_prediction=1.0,
        candidate_prediction=2.0,
        prediction_drift=1.0,
        objective_value=1.0,
        metadata={"foo": "bar"}
    )
    d = res.to_dict()
    assert d["source_id"] == "1"
    assert d["valid_rdkit"] is True
    assert d["metadata"]["foo"] == "bar"
