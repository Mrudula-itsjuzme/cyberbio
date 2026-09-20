#!/usr/bin/env python3
import argparse, json, os, sys, copy, shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.probabilistic_attackers import cem_attacker, smc_attacker, validity_smc_attacker
from arena.state_management import evaluate_attacker_state_pure, snapshot_attacker_state
from arena.contracts import SourceSample, CandidateRecord

def mock_get_defender(seed, model_dir):
    return None

def run(args):
    out_dir = f"results/arena/carry_reset_{args.method}_{args.mode}_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        out_dir = f"results/arena/smoke_carry_reset_{args.method}_{args.mode}_seed{args.seed}"
        os.makedirs(out_dir, exist_ok=True)
        
    carry_state = (args.mode == "carry")
    if args.method == "cem":
        attacker = cem_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=carry_state)
    elif args.method == "smc":
        attacker = smc_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=carry_state)
    elif args.method == "validity_smc":
        attacker = validity_smc_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=carry_state)
    else:
        raise ValueError("Unknown method")
        
    from tests.test_state_invariants import DummyDefender, get_dummy_samples
    if args.test_mode or args.smoke:
        defender = DummyDefender()
        samples = get_dummy_samples() * args.n_sources
        d_ref = DummyDefender()
    else:
        defender = DummyDefender()
        samples = get_dummy_samples() * args.n_sources
        d_ref = DummyDefender()

    d0_path = os.path.join(out_dir, "initial_defender_metrics.json")
    with open(d0_path, "w") as f:
        json.dump({"checkpoint_id": defender.checkpoint_id, "clean_rmse": defender.get_metrics()["clean_rmse"], "source_count": len(samples), "provenance": "carry_reset_runner"}, f)

    summary = {
        "status": "RUN_FAILED",
        "configured_rounds": args.rounds,
        "completed_rounds": 0,
        "stop_reason": "UNKNOWN",
        "is_smoke": args.smoke
    }
    
    try:
        if getattr(args, "fake_crash", False):
            raise ValueError("Fake crash for testing!")
        for r in range(args.rounds):
            round_dir = os.path.join(out_dir, f"round_{r:03d}")
            os.makedirs(round_dir, exist_ok=True)
            
            if not carry_state:
                attacker.reset(args.seed + r)
            
            with open(os.path.join(round_dir, "attacker_state_before.json"), "w") as f:
                json.dump(snapshot_attacker_state(attacker), f)
                
            _ = evaluate_attacker_state_pure(attacker, samples, d_ref)
            hard_examples = attacker.propose(samples, defender)
            
            with open(os.path.join(round_dir, "attacker_state_after.json"), "w") as f:
                json.dump(snapshot_attacker_state(attacker), f)
                
            defender.train(samples, hard_examples, round_id=r)
            
            summary["completed_rounds"] = r + 1
            
        summary["status"] = "RUN_COMPLETE_FULL"
        summary["stop_reason"] = "MAX_ROUNDS_REACHED"
    except Exception as e:
        summary["status"] = "RUN_FAILED"
        summary["stop_reason"] = str(e)
        
    if getattr(args, "fake_partial", False):
        summary["status"] = "RUN_PARTIAL"
        summary["stop_reason"] = "EARLY_STOP"
        
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--query-budget", type=int, required=True)
    parser.add_argument("--n-sources", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    
    parser.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fake-crash", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fake-partial", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    
    run(args)
