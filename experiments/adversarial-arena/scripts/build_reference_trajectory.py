#!/usr/bin/env python3
import os, sys, json, argparse, hashlib, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.attackers import random_smiles_adapter
from arena.transpolymer import TransPolymerDefender
from scripts.run_phase2_pilot import _load_sources, _load_clean_data
from arena.contracts import SourceSample
from arena.state_management import hash_model_state

def build_reference(seed, n_sources, rounds, budget):
    out_dir = f"results/arena/smoke_carry_reset_reference_seed{seed}" if args.smoke.lower() == "true" else f"results/arena/carry_reset_reference_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    ref_dir = os.path.join(out_dir, "reference_defender")
    os.makedirs(ref_dir, exist_ok=True)
    
    sources = _load_sources(seed, n_sources)
    source_ids = [s.source_id for s in sources]
    source_cohort_hash = hashlib.sha256(json.dumps(source_ids, sort_keys=True).encode()).hexdigest()
    with open(os.path.join(out_dir, "source_ids.json"), "w") as f:
        json.dump({"source_ids": source_ids, "hash": source_cohort_hash}, f)
        
    defender = TransPolymerDefender(
        project_root="../../materials-adversarial",
        seed=seed,
        checkpoint_dir="results/models/transformer_regressor",
        lambda_rep=0.5,
        learning_rate=0.001
    )
    
    model_hash = hash_model_state(defender)

    attacker = random_smiles_adapter(seed=seed, proposal_budget=budget)
    
    d0_metrics = defender.evaluate(sources)
    with open(os.path.join(out_dir, "initial_defender_metrics.json"), "w") as f:
        json.dump({
            "stage_index": 0,
            "checkpoint_id": defender.checkpoint_id,
            "checkpoint_sha256": model_hash,
            "clean_rmse": d0_metrics["rmse"],
            "source_count": len(sources),
            "source_cohort_hash": source_cohort_hash,
            "timestamp": time.time()
        }, f)
        
    d0_dir = os.path.join(ref_dir, "D_ref_000")
    os.makedirs(d0_dir, exist_ok=True)
    defender.checkpoint(os.path.join(d0_dir, "defender_checkpoint.json"))

    summary = {"status": "RUN_FAILED", "configured_rounds": rounds, "completed_rounds": 0, "stop_reason": "UNKNOWN", "stage_index": 0}
    try:
        for r in range(rounds):
            hard_examples = attacker.propose(sources, defender)
            clean_data = _load_clean_data(seed, 50)
            clean_samples = [SourceSample(source_id=s.source_id, input=s.input, target=s.target) for s in clean_data]
            
            defender.train(clean_samples, hard_examples, round_id=r)
            
            round_dir = os.path.join(ref_dir, f"D_ref_{r+1:03d}")
            os.makedirs(round_dir, exist_ok=True)
            defender.checkpoint(os.path.join(round_dir, "defender_checkpoint.json"))
            summary["completed_rounds"] = r + 1
            summary["stage_index"] = r + 1
            
        summary["status"] = "RUN_COMPLETE_FULL"
        summary["stop_reason"] = "MAX_ROUNDS_REACHED"
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["stop_reason"] = str(e)
        with open(os.path.join(out_dir, "reference_run_summary.json"), "w") as f:
            json.dump(summary, f)
        sys.exit(1)
    
    with open(os.path.join(out_dir, "reference_run_summary.json"), "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=str, default="true", help="True for smoke mode")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-sources", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--query-budget", type=int, required=True)
    args = parser.parse_args()
    build_reference(args.seed, args.n_sources, args.rounds, args.query_budget)
