import pandas as pd
import os

bc_df = pd.read_csv("../results/bio_cyber/aggregated_results.csv")
mat_df = pd.read_csv("../results/materials/aggregated_results.csv")

# Ensure domain columns exist
bc_df["domain"] = "bio_cyber"
if "domain" not in mat_df.columns:
    mat_df["domain"] = "materials"

cols = [
    "domain", "attack_condition", "search_budget",
    "mean_edit_distance", "mean_search_queries", "mean_attribution_queries",
    "mean_total_queries", "mean_runtime"
]

cross = pd.concat([bc_df, mat_df], ignore_index=True)

# Add domain specific metrics safely
if "mean_absolute_drift" in bc_df.columns:
    cross["bio_probability_drift"] = cross.apply(lambda r: r["mean_absolute_drift"] if r["domain"] == "bio_cyber" else pd.NA, axis=1)
if "success_rate" in bc_df.columns:
    cross["bio_label_flip_rate"] = cross.apply(lambda r: r["success_rate"] if r["domain"] == "bio_cyber" else pd.NA, axis=1)
if "mean_absolute_drift" in mat_df.columns:
    cross["materials_Tg_absolute_drift"] = cross.apply(lambda r: r["mean_absolute_drift"] if r["domain"] == "materials" else pd.NA, axis=1)
if "rdkit_validity_rate" in mat_df.columns:
    cross["materials_rdkit_validity_rate"] = cross.apply(lambda r: r["rdkit_validity_rate"] if r["domain"] == "materials" else pd.NA, axis=1)

os.makedirs("../results/cross_domain", exist_ok=True)
cross.to_csv("../results/cross_domain/attack_strategy_comparison.csv", index=False)
