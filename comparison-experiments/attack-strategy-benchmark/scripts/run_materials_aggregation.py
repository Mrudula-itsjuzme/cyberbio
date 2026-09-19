import pandas as pd
import numpy as np
import json
from scipy.stats import wilcoxon, binomtest

df = pd.read_csv("../results/materials/per_example_results.csv")
budgets = [5, 20, 50]
attacks = df["attack_condition"].unique()

agg_rows = []
for atk in attacks:
    for b in budgets:
        sub = df[(df["attack_condition"] == atk) & (df["search_budget"] == b)]
        if len(sub) == 0: continue
        
        n = len(sub)
        mean_abs_drift = sub["absolute_prediction_change"].mean()
        median_abs_drift = sub["absolute_prediction_change"].median()
        p90_abs_drift = sub["absolute_prediction_change"].quantile(0.9)
        
        mean_ed = sub["edit_distance"].mean()
        median_ed = sub["edit_distance"].median()
        
        mean_prop = sub["proposal_count"].mean()
        mean_sq = sub["search_queries"].mean()
        mean_aq = sub["attribution_queries"].mean()
        mean_tq = sub["total_queries"].mean()
        
        mean_rt = sub["runtime_seconds"].mean()
        median_rt = sub["runtime_seconds"].median()
        
        rdk_rate = (sub["rdkit_valid"] > 0).mean()
        can_rate = (sub["canonicalization_valid"] > 0).mean()
        tok_rate = (sub["tokenization_valid"] > 0).mean()
        
        agg_rows.append({
            "domain": "materials",
            "attack_condition": atk,
            "search_budget": b,
            "n": n,
            "mean_absolute_drift": mean_abs_drift,
            "median_absolute_drift": median_abs_drift,
            "p90_absolute_drift": p90_abs_drift,
            "mean_edit_distance": mean_ed,
            "median_edit_distance": median_ed,
            "mean_proposal_count": mean_prop,
            "mean_search_queries": mean_sq,
            "mean_attribution_queries": mean_aq,
            "mean_total_queries": mean_tq,
            "mean_runtime": mean_rt,
            "median_runtime": median_rt,
            "rdkit_validity_rate": rdk_rate,
            "canonicalization_validity_rate": can_rate,
            "tokenization_validity_rate": tok_rate
        })
        
agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv("../results/materials/aggregated_results.csv", index=False)

# Bootstrap CI func
np.random.seed(42)
N_BOOTSTRAP = 1000

def bootstrap_ci(diff_array, func=np.mean):
    if len(diff_array) == 0:
        return [np.nan, np.nan]
    boot_stats = []
    n = len(diff_array)
    for _ in range(N_BOOTSTRAP):
        sample = np.random.choice(diff_array, size=n, replace=True)
        boot_stats.append(func(sample))
    return [float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))]

# Stats
comparisons = [
    ("MCMC", "Random"),
    ("Evolutionary", "Random"),
    ("Attribution-High", "Attribution-Random"),
    ("Attribution-Low", "Attribution-Random")
]
stats = []

for b in budgets:
    sub = df[df["search_budget"] == b]
    for c1, c2 in comparisons:
        d1 = sub[sub["attack_condition"] == c1].sort_values("source_id")
        d2 = sub[sub["attack_condition"] == c2].sort_values("source_id")
        if len(d1) == 0 or len(d2) == 0: continue
        
        diff_drift = d1["absolute_prediction_change"].values - d2["absolute_prediction_change"].values
        p_drift = wilcoxon(d1["absolute_prediction_change"].values, d2["absolute_prediction_change"].values).pvalue if np.any(diff_drift) else np.nan
        dz_drift = np.mean(diff_drift) / np.std(diff_drift) if np.std(diff_drift) > 0 else np.nan
        ci_drift = bootstrap_ci(diff_drift, np.mean)
        
        diff_tq = d1["total_queries"].values - d2["total_queries"].values
        p_tq = wilcoxon(d1["total_queries"].values, d2["total_queries"].values).pvalue if np.any(diff_tq) else np.nan
        dz_tq = np.mean(diff_tq) / np.std(diff_tq) if np.std(diff_tq) > 0 else np.nan
        ci_tq = bootstrap_ci(diff_tq, np.mean)
        
        diff_ed = d1["edit_distance"].values - d2["edit_distance"].values
        p_ed = wilcoxon(d1["edit_distance"].values, d2["edit_distance"].values).pvalue if np.any(diff_ed) else np.nan
        dz_ed = np.mean(diff_ed) / np.std(diff_ed) if np.std(diff_ed) > 0 else np.nan
        ci_ed = bootstrap_ci(diff_ed, np.mean)
        
        stats.append({
            "comparison": f"{c1} vs {c2}",
            "budget": b,
            "n": len(d1),
            "metrics": {
                "absolute_prediction_change": {
                    "mean_paired_difference": float(np.mean(diff_drift)),
                    "cohen_dz": float(dz_drift),
                    "CI_lower": ci_drift[0],
                    "CI_upper": ci_drift[1],
                    "p_value": float(p_drift)
                },
                "total_queries": {
                    "mean_paired_difference": float(np.mean(diff_tq)),
                    "cohen_dz": float(dz_tq),
                    "CI_lower": ci_tq[0],
                    "CI_upper": ci_tq[1],
                    "p_value": float(p_tq)
                },
                "edit_distance": {
                    "mean_paired_difference": float(np.mean(diff_ed)),
                    "cohen_dz": float(dz_ed),
                    "CI_lower": ci_ed[0],
                    "CI_upper": ci_ed[1],
                    "p_value": float(p_ed)
                }
            }
        })
        
with open("../results/materials/statistical_results.json", "w") as f:
    json.dump(stats, f, indent=4)
    
with open("../results/materials/bootstrap_metadata.json", "w") as f:
    json.dump({"seed": 42, "iterations": N_BOOTSTRAP}, f, indent=4)
