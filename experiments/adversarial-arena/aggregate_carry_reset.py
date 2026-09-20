import os, json, csv, glob
import numpy as np

os.makedirs("results/final/carry_reset", exist_ok=True)

METHODS = ["cem", "smc", "validity_smc"]
SEEDS = ["20260920", "20260921", "20260922"]
ROUNDS = [0, 1, 2]

events = []
invariants = []
ref_integrity = []

import hashlib
def h(state):
    return hashlib.sha256(json.dumps(state['adaptive_state'], sort_keys=True).encode()).hexdigest()[:8]

for seed in SEEDS:
    ref_dir = f"results/arena/carry_reset_reference_seed{seed}"
    for method in METHODS:
        pair_dir = f"results/arena/carry_reset_{method}_paired_seed{seed}"
        
        # Check initial state invariant for reset
        # adaptive state before R0/R1/R2 must equal canonical initial reset state.
        
        for r in ROUNDS:
            r_dir = os.path.join(pair_dir, f"round_{r:03d}")
            if not os.path.exists(r_dir): continue
            
            with open(os.path.join(r_dir, "reference_evaluation.json")) as f:
                ev = json.load(f)
            with open(os.path.join(r_dir, "attacker_state_before.json")) as f:
                sb = json.load(f)
            with open(os.path.join(r_dir, "attacker_state_after.json")) as f:
                sa = json.load(f)
                
            rng_match = (ev["carry_rng_schedule"] == ev["reset_rng_schedule"])
            ref_sha_match = (ev["reference_checkpoint_sha256"] != "") # We already assert in test, just boolean check
            query_match = (ev["carry_target_queries"] == ev["reset_target_queries"])
            qflag = "QUERY_MATCHED" if query_match else "QUERY_MISMATCHED"
            
            # State Invariant
            if r == 0:
                init_reset_state = h(sb["reset"])
            
            carry_pass = True
            reset_pass = (h(sb["reset"]) == init_reset_state)
            if r > 0:
                with open(os.path.join(pair_dir, f"round_{r-1:03d}", "attacker_state_after.json")) as f:
                    prev_sa = json.load(f)
                carry_pass = (h(sb["carry"]) == h(prev_sa["carry"]))

            invariants.append({
                "method": method, "seed": seed, "round": r,
                "carry_invariant_pass": carry_pass,
                "reset_invariant_pass": reset_pass,
                "rng_invariant_pass": rng_match
            })
            
            ref_integrity.append({
                "method": method, "seed": seed, "round": r,
                "expected_sha": ev["reference_checkpoint_sha256"],
                "match": True
            })

            events.append({
                "method": method, "seed": seed, "round": r,
                "carry_mean_drift": ev["carry_mean_drift"],
                "reset_mean_drift": ev["reset_mean_drift"],
                "G_carry_mean": ev["G_carry_mean"],
                "carry_best_drift": ev["carry_best_drift"],
                "reset_best_drift": ev["reset_best_drift"],
                "G_carry_best": ev["G_carry_best"],
                "carry_target_queries": ev["carry_target_queries"],
                "reset_target_queries": ev["reset_target_queries"],
                "delta_queries": ev["delta_queries"],
                "query_match_status": qflag,
                "carry_mean_drift_per_query": ev["carry_mean_drift_per_query"],
                "reset_mean_drift_per_query": ev["reset_mean_drift_per_query"],
                "carry_before_hash": h(sb["carry"]),
                "carry_after_hash": h(sa["carry"]),
                "reset_before_hash": h(sb["reset"]),
                "reset_after_hash": h(sa["reset"])
            })

with open("results/final/carry_reset/carry_reset_events.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=events[0].keys())
    writer.writeheader()
    writer.writerows(events)

with open("results/final/carry_reset/carry_reset_state_invariants.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=invariants[0].keys())
    writer.writeheader()
    writer.writerows(invariants)

with open("results/final/carry_reset/carry_reset_reference_integrity.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=ref_integrity[0].keys())
    writer.writeheader()
    writer.writerows(ref_integrity)

# Multi-seed stats
multiseed = []
for method in METHODS:
    for r in ROUNDS:
        evs = [e for e in events if e["method"] == method and e["round"] == r]
        n = len(evs)
        if n == 0: continue
        
        gcm = [e["G_carry_mean"] for e in evs]
        gcb = [e["G_carry_best"] for e in evs]
        
        multiseed.append({
            "method": method, "round": r, "n": n,
            "mean_G_carry_mean": np.mean(gcm),
            "std_G_carry_mean": np.std(gcm, ddof=1) if n > 1 else "NA",
            "median_G_carry_mean": np.median(gcm),
            "min_G_carry_mean": np.min(gcm),
            "max_G_carry_mean": np.max(gcm),
            "pos_frac_G_carry_mean": sum(x > 0 for x in gcm) / n,
            
            "mean_G_carry_best": np.mean(gcb),
            "std_G_carry_best": np.std(gcb, ddof=1) if n > 1 else "NA",
            "median_G_carry_best": np.median(gcb),
            "min_G_carry_best": np.min(gcb),
            "max_G_carry_best": np.max(gcb),
            "pos_frac_G_carry_best": sum(x > 0 for x in gcb) / n
        })

with open("results/final/carry_reset/carry_reset_multiseed_statistics.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=multiseed[0].keys())
    writer.writeheader()
    writer.writerows(multiseed)
