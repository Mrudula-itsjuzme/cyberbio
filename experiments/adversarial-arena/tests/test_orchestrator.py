import sys, os, json, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RUNNER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "run_carry_reset.py"))

def run_orchestrator(extra_args):
    cmd = [sys.executable, RUNNER, "--method", "cem", "--mode", "carry", "--seed", "42", "--rounds", "2", "--query-budget", "5", "--n-sources", "1", "--test-mode"] + extra_args
    subprocess.run(cmd, check=False)
    
def test_orchestrator_persists_d0_before_round0():
    run_orchestrator([])
    out_dir = "results/arena/carry_reset_cem_carry_seed42"
    d0_path = os.path.join(out_dir, "initial_defender_metrics.json")
    
    assert os.path.exists(d0_path), "D0 metrics not persisted"
    with open(d0_path) as f:
        d0 = json.load(f)
    assert "clean_rmse" in d0
    assert "checkpoint_id" in d0

def test_full_completion_status():
    run_orchestrator([])
    out_dir = "results/arena/carry_reset_cem_carry_seed42"
    with open(os.path.join(out_dir, "run_summary.json")) as f:
        res = json.load(f)
    assert res["status"] == "RUN_COMPLETE_FULL"
    assert res["completed_rounds"] == 2

def test_partial_completion_status():
    run_orchestrator(["--fake-partial"])
    out_dir = "results/arena/carry_reset_cem_carry_seed42"
    with open(os.path.join(out_dir, "run_summary.json")) as f:
        res = json.load(f)
    assert res["status"] == "RUN_PARTIAL"

def test_failed_status():
    run_orchestrator(["--fake-crash"])
    out_dir = "results/arena/carry_reset_cem_carry_seed42"
    with open(os.path.join(out_dir, "run_summary.json")) as f:
        res = json.load(f)
    assert res["status"] == "RUN_FAILED"

if __name__ == "__main__":
    print("Testing orchestrator...")
    test_orchestrator_persists_d0_before_round0()
    test_full_completion_status()
    test_partial_completion_status()
    test_failed_status()
    print("ALL TESTS PASSED")
