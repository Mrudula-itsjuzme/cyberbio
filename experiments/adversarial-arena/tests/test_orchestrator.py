import json, os
from arena.state_management import hash_model_state

def get_smoke_dir(method):
    return f"results/arena/smoke_carry_reset_{method}_paired_seed20260920"

def get_ref_dir():
    return "results/arena/smoke_carry_reset_reference_seed20260920"

def test_query_accounting_artifact_derived():
    for method in ["cem", "smc", "validity_smc"]:
        for r in range(2):
            with open(os.path.join(get_smoke_dir(method), f"round_{r:03d}", "reference_evaluation.json")) as f:
                ev = json.load(f)
            assert ev["carry_target_queries"] <= 5
            assert ev["reset_target_queries"] <= 5
            assert ev["delta_queries"] == ev["carry_target_queries"] - ev["reset_target_queries"]
            assert ev["delta_queries"] == 0

def test_d0_ordering():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "initial_defender_metrics.json")) as f:
        d0 = json.load(f)
    with open(os.path.join(d, "run_summary.json")) as f:
        summary = json.load(f)
    assert d0["stage_index"] == 0
    assert summary["stage_index"] > 0
    assert "checkpoint_sha256" in d0

def test_reference_checkpoint_identity():
    # Verify the reference checkpoint SHA matches the actual loaded weights on disk for D0 and D1
    from arena.transpolymer import TransPolymerDefender
    
    # Check D_ref_000
    ref0_path = os.path.join(get_ref_dir(), "reference_defender", "D_ref_000", "defender_checkpoint.json")
    assert os.path.exists(ref0_path)
    m = TransPolymerDefender('../../materials-adversarial', seed=123, checkpoint_dir='results/models/transformer_regressor')
    m.load_checkpoint(ref0_path)
    d0_hash = hash_model_state(m)
    with open(os.path.join(get_smoke_dir("cem"), "round_000", "reference_evaluation.json")) as f:
        ev = json.load(f)
    assert d0_hash == ev["reference_checkpoint_sha256"]
    
    # Check D_ref_001
    ref1_path = os.path.join(get_ref_dir(), "reference_defender", "D_ref_001", "defender_checkpoint.json")
    assert os.path.exists(ref1_path)
    m = TransPolymerDefender('../../materials-adversarial', seed=123, checkpoint_dir='results/models/transformer_regressor')
    m.load_checkpoint(ref1_path)
    d1_hash = hash_model_state(m)
    with open(os.path.join(get_smoke_dir("cem"), "round_001", "reference_evaluation.json")) as f:
        ev = json.load(f)
    assert d1_hash == ev["reference_checkpoint_sha256"]

def test_initial_model_identity():
    # Verify reference initial D0 hash == paired-run initial D0 hash
    with open(os.path.join(get_ref_dir(), "initial_defender_metrics.json")) as f:
        ref_d0 = json.load(f)
    with open(os.path.join(get_smoke_dir("cem"), "initial_defender_metrics.json")) as f:
        pair_d0 = json.load(f)
    assert ref_d0["checkpoint_sha256"] == pair_d0["checkpoint_sha256"]

def test_carry_reset_rng_schedule_identical():
    for method in ["cem", "smc", "validity_smc"]:
        for r in range(2):
            d = get_smoke_dir(method)
            with open(os.path.join(d, f"round_{r:03d}", "reference_evaluation.json")) as f:
                ev = json.load(f)
            assert ev["carry_rng_schedule"] == ev["reset_rng_schedule"]
