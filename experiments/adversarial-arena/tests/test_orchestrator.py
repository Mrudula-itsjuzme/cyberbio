import json, os

def get_smoke_dir(method):
    return f"results/arena/smoke_carry_reset_{method}_paired_seed20260920"

def test_query_budget_matched():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "round_000", "reference_evaluation.json")) as f:
        ev = json.load(f)
    assert ev["carry_target_queries"] <= ev["configured_query_budget"]
    assert ev["reset_target_queries"] <= ev["configured_query_budget"]

def test_full_completion_status():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "run_summary.json")) as f:
        summary = json.load(f)
    assert summary["status"] == "RUN_COMPLETE_FULL"
    assert summary.get("SMOKE_ONLY_NON_CANONICAL") is True

def test_orchestrator_persists_d0_before_round0():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "initial_defender_metrics.json")) as f:
        d0 = json.load(f)
    assert "checkpoint_id" in d0
    assert "checkpoint_hash" in d0
    assert "clean_rmse" in d0
    assert "source_cohort_hash" in d0
import subprocess
def test_failed_run_returns_nonzero():
    # Calling the script with a missing reference trajectory should fail
    # We pass an unknown seed
    cmd = ["/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python", "scripts/run_carry_reset_pair.py", "--method", "cem", "--seed", "999999", "--rounds", "1", "--query-budget", "5", "--n-sources", "1"]
    res = subprocess.run(cmd, check=False, capture_output=True)
    assert res.returncode != 0
