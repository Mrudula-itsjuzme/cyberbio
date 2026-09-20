import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path("/home/mrudula/Downloads/DL_cyberbio/.phase5b-worktree/materials-adversarial")
OUT = ROOT / "results/invariance_defense"

# 1. Tokenizer contribution (I will extract this from the results)
# 6. Positional ablation
# 9. Seed stability
# 11. Residual failures

def run():
    print("Running audit...")
    
    # Let's load the model comparison and stats
    clean = pd.read_csv(OUT / "clean_metrics.csv")
    repeat = pd.read_csv(OUT / "repeat_shift_robustness.csv")
    broad = pd.read_csv(OUT / "broader_equivalence_robustness.csv")
    families = pd.read_csv(OUT / "family_robustness.csv")
    
    # 6. Positional Ablation
    pos_model = "broader_average_no_position_seed20260815"
    baseline = "baseline"
    
    pos_clean = clean[clean.model == pos_model].iloc[0]
    base_clean = clean[clean.model == baseline].iloc[0]
    
    pos_repeat = repeat[repeat.model == pos_model].iloc[0]
    base_repeat = repeat[repeat.model == baseline].iloc[0]
    
    pos_broad = broad[broad.model == pos_model].iloc[0]
    base_broad = broad[broad.model == baseline].iloc[0]
    
    branch = families[families.family == "branch_order_permutation"]
    pos_branch = branch[branch.model == pos_model].iloc[0]
    base_branch = branch[branch.model == baseline].iloc[0]
    
    pos_rows = [
        {
            "model": "baseline",
            "seed": "historical",
            "positional_encoding": "yes",
            "clean_mae": base_clean.mae,
            "clean_rmse": base_clean.rmse,
            "repeat_shift_worst_mean": base_repeat.mean_source_worst,
            "broader_worst_mean": base_broad.mean_source_worst,
            "branch_order_mean_drift": base_branch.mean_source_worst
        },
        {
            "model": "broader_average_no_position",
            "seed": "20260815",
            "positional_encoding": "no",
            "clean_mae": pos_clean.mae,
            "clean_rmse": pos_clean.rmse,
            "repeat_shift_worst_mean": pos_repeat.mean_source_worst,
            "broader_worst_mean": pos_broad.mean_source_worst,
            "branch_order_mean_drift": pos_branch.mean_source_worst
        }
    ]
    pd.DataFrame(pos_rows).to_csv(OUT / "positional_ablation.csv", index=False)
    
    # 9. Seed stability
    seeds = [20260815, 20260816, 20260817]
    models = ["ordinary_augmentation", "repeat_shift_average", "broader_average", "broader_worst"]
    stability = []
    for model in models:
        for seed in seeds:
            name = f"{model}_seed{seed}"
            c = clean[clean.model == name].iloc[0]
            r = repeat[repeat.model == name].iloc[0]
            b = broad[broad.model == name].iloc[0]
            stability.append({
                "model": model,
                "seed": seed,
                "clean_mae": c.mae,
                "repeat_shift_worst_mean": r.mean_source_worst,
                "broader_worst_mean": b.mean_source_worst
            })
    pd.DataFrame(stability).to_csv(OUT / "seed_stability.csv", index=False)
    
    print("Seed stability summary:")
    st_df = pd.DataFrame(stability)
    for model in models:
        df_m = st_df[st_df.model == model]
        print(f"{model}:")
        print(f"  clean_mae: mean={df_m.clean_mae.mean():.5f}, std={df_m.clean_mae.std():.5f}, min={df_m.clean_mae.min():.5f}, max={df_m.clean_mae.max():.5f}")
        print(f"  broader_worst: mean={df_m.broader_worst_mean.mean():.5f}, std={df_m.broader_worst_mean.std():.5f}, min={df_m.broader_worst_mean.min():.5f}, max={df_m.broader_worst_mean.max():.5f}")

    print("Success")

if __name__ == "__main__":
    run()
