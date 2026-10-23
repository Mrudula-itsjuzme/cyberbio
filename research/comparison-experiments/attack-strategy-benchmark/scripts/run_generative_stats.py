import json, os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent

def compute_stats(domain):
    results_dir = ROOT / f"results/generative/{domain}"
    
    gen_df = pd.read_csv(results_dir / "per_example_results.csv")
    class_df = pd.read_csv(ROOT / f"results/{domain}/per_example_results.csv")
    
    stats_out = {}
    
    if domain == "bio_cyber":
        metrics_map = {
            "absolute_drift": "absolute_prediction_change",
            "edit_distance": "edit_distance",
            "attack_model_queries": "search_queries",
            "runtime_seconds": "runtime_seconds"
        }
    else:
        metrics_map = {
            "absolute_drift": "absolute_prediction_change",
            "edit_distance": "edit_distance",
            "attack_model_queries": "attack_model_queries",
            "proposal_attempts": "proposal_attempts",
            "runtime_seconds": "runtime_seconds"
        }
    
    for budget in [5, 20, 50]:
        budget_stats = {}
        for base in ["Random", "MCMC", "Evolutionary"]:
            pair_stats = {}
            for gen_metric, class_metric in metrics_map.items():
                g_vals = gen_df[(gen_df["attack_condition"] == "Generative") & (gen_df["search_budget"] == budget)][gen_metric].values
                b_vals = class_df[(class_df["attack_condition"] == base) & (class_df["search_budget"] == budget)][class_metric].values
                
                if len(g_vals) == len(b_vals) and len(g_vals) > 0:
                    diff = g_vals - b_vals
                    if np.all(diff == 0):
                        pval = 1.0
                    else:
                        try:
                            _, pval = wilcoxon(g_vals, b_vals)
                        except:
                            pval = 1.0
                    pair_stats[gen_metric] = {"Generative_mean": np.nanmean(g_vals), f"{base}_mean": np.nanmean(b_vals), "p_value": pval}
            budget_stats[f"Generative_vs_{base}"] = pair_stats
        stats_out[str(budget)] = budget_stats
        
    with open(results_dir / "statistical_results.json", "w") as f:
        json.dump(stats_out, f, indent=2)

if __name__ == "__main__":
    compute_stats("bio_cyber")
    compute_stats("materials")
