#!/usr/bin/env python3
"""Compile Phase 7 Adaptive Attack Results."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

def bootstrap_ci(diff_array, ids_array, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(ids_array)
    n_ids = len(unique_ids)
    
    means = []
    for _ in range(n_boot):
        sample_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        # For simplicity, if ids are 1:1 with diff_array:
        sample_diffs = []
        for sid in sample_ids:
            idx = np.where(ids_array == sid)[0]
            if len(idx) > 0:
                sample_diffs.append(diff_array[idx[0]])
        means.append(np.mean(sample_diffs))
        
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def compile_report():
    repo_root = Path(__file__).resolve().parent.parent
    search_dir = repo_root / "results" / "phase7_adaptive_search"
    search_results_path = search_dir / "search_trajectories.jsonl"
    transfer_results_path = search_dir / "transfer_evaluation.json"
    
    if not search_results_path.exists():
        print("Run search first.")
        return
        
    records = []
    with search_results_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    df = pd.DataFrame(records)
    
    report = {
        "budget_analysis": {},
        "strategy_comparison_budget_50": {},
        "defense_resistance_budget_50": {},
        "transfer_analysis": {}
    }
    
    # Analyze by budget and model
    for budget in df["budget"].unique():
        bdf = df[df["budget"] == budget]
        b_res = {}
        for model in bdf["model"].unique():
            mdf = bdf[bdf["model"] == model]
            s_res = {}
            for strategy in mdf["strategy"].unique():
                sdf = mdf[mdf["strategy"] == strategy]
                s_res[strategy] = {
                    "mean_drift": float(sdf["best_drift"].mean()),
                    "stress_success_rate": float((sdf["best_drift"] > 0.4862).mean()),
                    "mean_queries": float(sdf["candidate_model_queries"].mean()),
                    "valid_proposals": float(sdf["valid_candidates"].mean())
                }
            b_res[model] = s_res
        report["budget_analysis"][int(budget)] = b_res
        
    # Strategy Comparison on Baseline at Budget 50
    df50_base = df[(df["budget"] == 50) & (df["model"] == "ordinary_baseline")]
    ids = df50_base[df50_base["strategy"] == "random"]["source_id"].values
    
    strat_drifts = {}
    for s in ["random", "greedy", "metropolis_mcmc"]:
        # align by id
        s_df = df50_base[df50_base["strategy"] == s].set_index("source_id").loc[ids]
        strat_drifts[s] = s_df["best_drift"].values
        
    pairs = [("greedy", "random"), ("metropolis_mcmc", "random"), ("metropolis_mcmc", "greedy")]
    for s1, s2 in pairs:
        diff = strat_drifts[s1] - strat_drifts[s2]
        ci_lower, ci_upper = bootstrap_ci(diff, ids)
        report["strategy_comparison_budget_50"][f"{s1}_vs_{s2}"] = {
            "mean_difference": float(np.mean(diff)),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper)
        }
        
    # Defense Resistance at Budget 50
    for strat in ["random", "greedy", "metropolis_mcmc"]:
        df50_strat = df[(df["budget"] == 50) & (df["strategy"] == strat)]
        base_df = df50_strat[df50_strat["model"] == "ordinary_baseline"].set_index("source_id")
        robust_df = df50_strat[df50_strat["model"] == "mixed_robust"].set_index("source_id")
        
        common_ids = base_df.index.intersection(robust_df.index).values
        diff = robust_df.loc[common_ids, "best_drift"].values - base_df.loc[common_ids, "best_drift"].values
        
        ci_lower, ci_upper = bootstrap_ci(diff, common_ids)
        report["defense_resistance_budget_50"][strat] = {
            "robust_vs_baseline_mean_diff": float(np.mean(diff)),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper)
        }
        
    # Transfer Analysis
    if transfer_results_path.exists():
        with transfer_results_path.open("r", encoding="utf-8") as f:
            trans_recs = json.load(f)
            
        tdf = pd.DataFrame(trans_recs)
        for s in tdf["strategy"].unique():
            report["transfer_analysis"][s] = {}
            sdf = tdf[tdf["strategy"] == s]
            
            # Source: Baseline -> Eval: Robust
            b_to_r = sdf[(sdf["source_model"] == "ordinary_baseline") & (sdf["eval_model"] == "mixed_robust")]
            if len(b_to_r) > 0:
                report["transfer_analysis"][s]["baseline_to_robust_mean_drift"] = float(b_to_r["transfer_drift"].mean())
                report["transfer_analysis"][s]["baseline_to_robust_target_drift"] = float(b_to_r["target_drift"].mean())
                
            r_to_b = sdf[(sdf["source_model"] == "mixed_robust") & (sdf["eval_model"] == "ordinary_baseline")]
            if len(r_to_b) > 0:
                report["transfer_analysis"][s]["robust_to_baseline_mean_drift"] = float(r_to_b["transfer_drift"].mean())
                report["transfer_analysis"][s]["robust_to_baseline_target_drift"] = float(r_to_b["target_drift"].mean())

    with (search_dir / "compiled_report.json").open("w") as f:
        json.dump(report, f, indent=2)
        
    print("Report compiled.")

if __name__ == "__main__":
    compile_report()
