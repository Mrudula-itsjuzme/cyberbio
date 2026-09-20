#!/usr/bin/env python3
import os, sys, json, argparse, hashlib, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.probabilistic_attackers import cem_attacker, smc_attacker, validity_smc_attacker
from arena.state_management import evaluate_attacker_state_pure, snapshot_attacker_state, hash_adaptive_state, hash_full_state, derive_seed, hash_model_state
from arena.contracts import SourceSample
from arena.transpolymer import TransPolymerDefender
from scripts.run_phase2_pilot import _load_sources, _load_clean_data

def run_paired(args):
    out_dir = f"results/arena/smoke_carry_reset_{args.method}_paired_seed{args.seed}" if args.smoke.lower() == "true" else f"results/arena/carry_reset_{args.method}_paired_seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    
    ref_dir = f"results/arena/smoke_carry_reset_reference_seed{args.seed}" if args.smoke.lower() == "true" else f"results/arena/carry_reset_reference_seed{args.seed}"
    with open(os.path.join(ref_dir, "source_ids.json")) as f:
        ref_cohort = json.load(f)
        
    sources = _load_sources(args.seed, args.n_sources)
    source_ids = [s.source_id for s in sources]
    source_cohort_hash = hashlib.sha256(json.dumps(source_ids, sort_keys=True).encode()).hexdigest()
    
    if source_cohort_hash != ref_cohort["hash"]:
        print("Cohort hash mismatch!")
        sys.exit(1)
    
    if args.method == "cem":
        A_carry = cem_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=True)
        A_reset = cem_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=False)
    elif args.method == "smc":
        A_carry = smc_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=True)
        A_reset = smc_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=False)
    elif args.method == "validity_smc":
        A_carry = validity_smc_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=True)
        A_reset = validity_smc_attacker(seed=args.seed, query_budget=args.query_budget, carry_state=False)
    else:
        raise ValueError("Unknown method")
        
    D_ref = TransPolymerDefender(
        project_root="../../materials-adversarial",
        seed=args.seed,
        checkpoint_dir="results/models/transformer_regressor",
        lambda_rep=0.5,
        learning_rate=0.001
    )
    
    model_hash = hash_model_state(D_ref)
        
    d0_metrics = D_ref.evaluate(sources)
    with open(os.path.join(out_dir, "initial_defender_metrics.json"), "w") as f:
        json.dump({
            "stage_index": 0,
            "checkpoint_id": D_ref.checkpoint_id,
            "checkpoint_sha256": model_hash,
            "clean_rmse": d0_metrics["rmse"],
            "source_count": len(sources),
            "source_cohort_hash": source_cohort_hash,
            "timestamp": time.time()
        }, f)
        
    summary = {"status": "RUN_FAILED", "configured_rounds": args.rounds, "completed_rounds": 0, "stop_reason": "UNKNOWN"}
    
    try:
        for r in range(args.rounds):
            round_dir = os.path.join(out_dir, f"round_{r:03d}")
            os.makedirs(round_dir, exist_ok=True)
            
            ref_ckpt = os.path.join(ref_dir, "reference_defender", f"D_ref_{r:03d}", "defender_checkpoint.json")
            if not os.path.exists(ref_ckpt):
                raise FileNotFoundError(f"Missing reference checkpoint {ref_ckpt}")
            D_ref.load_checkpoint(ref_ckpt)
            
            round_model_hash = hash_model_state(D_ref)

            # Build explicit RNG schedule for this round
            rng_schedule = [derive_seed(args.seed, r, s.source_id, "proposal") for s in sources]
            with open(os.path.join(round_dir, "rng_schedule.json"), "w") as f:
                json.dump({"experiment_round": r, "seeds": rng_schedule}, f)

            # Assign schedule
            A_reset.reset(args.seed)
            A_carry._rng_schedule = list(rng_schedule)
            A_reset._rng_schedule = list(rng_schedule)
            
            with open(os.path.join(round_dir, "attacker_state_before.json"), "w") as f:
                json.dump({
                    "carry": snapshot_attacker_state(A_carry),
                    "reset": snapshot_attacker_state(A_reset)
                }, f)
                
            eval_carry = evaluate_attacker_state_pure(A_carry, sources, D_ref)
            eval_reset = evaluate_attacker_state_pure(A_reset, sources, D_ref)
            
            mc = eval_carry["metrics"].get("mean_drift", 0.0)
            bc = eval_carry["metrics"].get("best_drift", 0.0)
            mr = eval_reset["metrics"].get("mean_drift", 0.0)
            br = eval_reset["metrics"].get("best_drift", 0.0)
            
            qc = eval_carry["metrics"].get("target_model_queries", 0)
            qr = eval_reset["metrics"].get("target_model_queries", 0)
            
            with open(os.path.join(round_dir, "reference_evaluation.json"), "w") as f:
                json.dump({
                    "method": args.method, "seed": args.seed, "round": r,
                    "reference_checkpoint_id": D_ref.checkpoint_id,
                    "reference_checkpoint_sha256": round_model_hash,
                    "carry_rng_schedule": rng_schedule, "reset_rng_schedule": rng_schedule,
                    "carry_mean_drift": mc, "reset_mean_drift": mr,
                    "carry_best_drift": bc, "reset_best_drift": br,
                    "G_carry_mean": mc - mr, "G_carry_best": bc - br,
                    "carry_target_queries": qc,
                    "reset_target_queries": qr,
                    "configured_query_budget": args.query_budget * len(sources),
                    "delta_queries": qc - qr,
                    "carry_mean_drift_per_query": mc / qc if qc else 0,
                    "reset_mean_drift_per_query": mr / qr if qr else 0
                }, f)
                
            # Progress carry state
            A_carry._rng_schedule = list(rng_schedule) # re-supply for progression
            _ = A_carry.propose(sources, D_ref)
            
            with open(os.path.join(round_dir, "attacker_state_after.json"), "w") as f:
                json.dump({
                    "carry": snapshot_attacker_state(A_carry),
                    "reset": snapshot_attacker_state(A_reset)
                }, f)
            
            summary["completed_rounds"] = r + 1
            summary["stage_index"] = r + 1
            
        summary["status"] = "RUN_COMPLETE_FULL"
        summary["stop_reason"] = "MAX_ROUNDS_REACHED"
        summary["SMOKE_ONLY_NON_CANONICAL"] = (args.smoke.lower() == "true")
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["status"] = "RUN_FAILED"
        summary["stop_reason"] = str(e)
        with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f)
        sys.exit(1)
    
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--smoke", type=str, default="true", help="True for smoke mode")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--query-budget", type=int, required=True)
    parser.add_argument("--n-sources", type=int, required=True)
    args = parser.parse_args()
    run_paired(args)
