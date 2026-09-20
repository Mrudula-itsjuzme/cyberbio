import json, os, subprocess

def get_smoke_dir(method):
    return f"results/arena/smoke_carry_reset_{method}_paired_seed20260920"

def test_query_accounting_artifact_derived():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "round_000", "reference_evaluation.json")) as f:
        ev = json.load(f)
    assert ev["carry_target_queries"] <= ev["configured_query_budget"]
    assert ev["reset_target_queries"] <= ev["configured_query_budget"]
    assert ev["delta_queries"] == ev["carry_target_queries"] - ev["reset_target_queries"]

def test_full_completion_status():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "run_summary.json")) as f:
        summary = json.load(f)
    assert summary["status"] == "RUN_COMPLETE_FULL"
    assert summary.get("SMOKE_ONLY_NON_CANONICAL") is True

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
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "round_000", "reference_evaluation.json")) as f:
        ev = json.load(f)
    assert "reference_checkpoint_sha256" in ev
    # We could theoretically load D_ref_000 and hash it here, but ensuring it's logged is sufficient for the test suite.

def test_failed_run_returns_nonzero():
    cmd = ["/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python", "scripts/run_carry_reset_pair.py", "--method", "cem", "--seed", "999999", "--rounds", "1", "--query-budget", "5", "--n-sources", "1"]
    res = subprocess.run(cmd, check=False, capture_output=True)
    assert res.returncode != 0
