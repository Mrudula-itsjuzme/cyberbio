import os, json
import pandas as pd
import numpy as np
import hashlib
import sys
sys.path.insert(0, "/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/src")
from arena.transpolymer import TransPolymerDefender
from arena.state_management import hash_model_state

def h(state_dict):
    return hashlib.sha256(json.dumps(state_dict, sort_keys=True).encode()).hexdigest()

METHODS = ["cem", "smc", "validity_smc"]
SEEDS = ["20260920", "20260921", "20260922"]
ROUNDS = [0, 1, 2]

os.makedirs("results/final/carry_reset", exist_ok=True)

# Rebuild all tables from raw artifacts
events = []
invariants = []
ref_integrity = []
d0_identity = []

# Reference SHAs
ref_shas = {}
for seed in SEEDS:
    for r in ROUNDS:
        ckpt_path = f"results/arena/carry_reset_reference_seed{seed}/reference_defender/D_ref_{r:03d}/defender_checkpoint.json"
        if os.path.exists(ckpt_path):
            try:
                defender = TransPolymerDefender(project_root="../../materials-adversarial", seed=int(seed), checkpoint_dir="/tmp")
                defender.load_checkpoint(ckpt_path)
                ref_shas[(seed, r)] = hash_model_state(defender)
            except Exception as e:
                ref_shas[(seed, r)] = f"ERROR:{e}"

# D0 identity
for seed in SEEDS:
    ref_d0_path = f"results/arena/carry_reset_reference_seed{seed}/initial_defender_metrics.json"
    ref_d0_sha = "MISSING_REF"
    if os.path.exists(ref_d0_path):
        with open(ref_d0_path) as f:
            ref_d0_sha = json.load(f).get("checkpoint_sha256", "MISSING")
    for method in METHODS:
        pair_d0_path = f"results/arena/carry_reset_{method}_paired_seed{seed}/initial_defender_metrics.json"
        pair_d0_sha = "MISSING_PAIR"
        if os.path.exists(pair_d0_path):
            with open(pair_d0_path) as f:
                pair_d0_sha = json.load(f).get("checkpoint_sha256", "MISSING")
        d0_identity.append({
            "method": method, "seed": seed,
            "reference_d0_sha": ref_d0_sha, "paired_d0_sha": pair_d0_sha,
            "match": (ref_d0_sha == pair_d0_sha)
        })

# Events + invariants + ref integrity
for seed in SEEDS:
    init_reset_state_hash = None
    for method in METHODS:
        pair_dir = f"results/arena/carry_reset_{method}_paired_seed{seed}"
        for r in ROUNDS:
            r_dir = os.path.join(pair_dir, f"round_{r:03d}")
            if not os.path.exists(r_dir):
                continue
            with open(os.path.join(r_dir, "reference_evaluation.json")) as f:
                ev = json.load(f)
            with open(os.path.join(r_dir, "attacker_state_before.json")) as f:
                sb = json.load(f)
            with open(os.path.join(r_dir, "attacker_state_after.json")) as f:
                sa = json.load(f)

            if r == 0 and method == METHODS[0]:
                init_reset_state_hash = h(sb["reset"]["adaptive_state"])

            reset_pass = (h(sb["reset"]["adaptive_state"]) == init_reset_state_hash) if init_reset_state_hash else True
            carry_pass = True
            if r > 0:
                prev_r_dir = os.path.join(pair_dir, f"round_{r-1:03d}")
                with open(os.path.join(prev_r_dir, "attacker_state_after.json")) as f:
                    prev_sa = json.load(f)
                carry_pass = (h(sb["carry"]["adaptive_state"]) == h(prev_sa["carry"]["adaptive_state"]))
            rng_match = (ev["carry_rng_schedule"] == ev["reset_rng_schedule"])

            obs_sha = ev.get("reference_checkpoint_sha256", "MISSING")
            expected_sha = ref_shas.get((seed, r), "MISSING")
            sha_match = (obs_sha == expected_sha)
            ref_integrity.append({
                "method": method, "seed": seed, "round": r,
                "expected_sha": expected_sha, "observed_sha": obs_sha, "match": sha_match
            })
            invariants.append({
                "method": method, "seed": seed, "round": r,
                "carry_invariant_pass": carry_pass,
                "reset_invariant_pass": reset_pass,
                "rng_invariant_pass": rng_match
            })
            qmatch = (ev["carry_target_queries"] == ev["reset_target_queries"])
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
                "delta_queries": ev.get("delta_queries", ev["carry_target_queries"] - ev["reset_target_queries"]),
                "query_match_status": "QUERY_MATCHED" if qmatch else "QUERY_MISMATCHED",
                "carry_mean_drift_per_query": ev["carry_mean_drift_per_query"],
                "reset_mean_drift_per_query": ev["reset_mean_drift_per_query"],
                "delta_mean_drift_per_query": ev["carry_mean_drift_per_query"] - ev["reset_mean_drift_per_query"]
            })

df = pd.DataFrame(events)
df.to_csv("results/final/carry_reset/carry_reset_events_v2.csv", index=False)

matched = df[df["query_match_status"] == "QUERY_MATCHED"]
matched.to_csv("results/final/carry_reset/carry_reset_primary_matched.csv", index=False)

df.to_csv("results/final/carry_reset/carry_reset_secondary_all_events.csv", index=False)

df[["method","seed","round","carry_mean_drift_per_query","reset_mean_drift_per_query","delta_mean_drift_per_query"]].to_csv("results/final/carry_reset/carry_reset_per_query.csv", index=False)

# Per-seed post-init summary
rows = []
for m in METHODS:
    for s in SEEDS:
        sub = df[(df["method"] == m) & (df["seed"] == s) & (df["round"] > 0)]
        rows.append({
            "method": m, "seed": s,
            "postinit_mean_G_carry": sub["G_carry_mean"].mean() if len(sub) > 0 else float("nan"),
            "n_rounds": len(sub)
        })
pd.DataFrame(rows).to_csv("results/final/carry_reset/carry_reset_postinit_seed_summary.csv", index=False)

pd.DataFrame(d0_identity).to_csv("results/final/carry_reset/carry_reset_d0_identity.csv", index=False)
pd.DataFrame(ref_integrity).to_csv("results/final/carry_reset/carry_reset_reference_integrity_v2.csv", index=False)
pd.DataFrame(invariants).to_csv("results/final/carry_reset/carry_reset_state_invariants_v2.csv", index=False)
print("All tables written.")
