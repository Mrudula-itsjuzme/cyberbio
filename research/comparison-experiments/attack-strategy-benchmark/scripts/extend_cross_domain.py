import pandas as pd
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

out_path = ROOT / "results/cross_domain/attack_strategy_comparison.csv"
df_main = pd.read_csv(out_path)

# Remove any existing Generative rows
df_main = df_main[~df_main["attack_condition"].str.contains("Generative")]

new_rows = []

# bio_cyber generative
bc_gen = pd.read_csv(ROOT / "results/generative/bio_cyber/aggregated_results.csv")
for _, row in bc_gen.iterrows():
    cond = row["attack_condition"]
    budget = row["search_budget"]
    new_rows.append({
        "attack_condition": cond,
        "mean_runtime": row["runtime_seconds"],
        "mean_edit_distance": row["edit_distance"],
        "domain": "bio_cyber",
        "bio_cyber_label_flip_rate": row.get("success", np.nan),
        "bio_cyber_probability_drift": row["absolute_drift"],
        "materials_Tg_absolute_drift": np.nan,
        "materials_rdkit_proposal_validity_rate": np.nan,
        "mean_total_model_queries": row["total_model_queries"],
        "search_budget": budget,
        "mean_attack_model_queries": row["attack_model_queries"],
        "mean_attribution_queries": 0.0,
        "mean_proposal_attempts": row["proposal_attempts"]
    })
    
# materials generative
mat_gen = pd.read_csv(ROOT / "results/generative/materials/aggregated_results.csv")
for _, row in mat_gen.iterrows():
    cond = row["attack_condition"]
    budget = row["search_budget"]
    # Calculate validity rate from valid_cands / proposal_attempts if available in original file
    # We can approximate with valid_candidates / proposal_attempts
    val_rate = row["valid_candidates"] / row["proposal_attempts"] if row["proposal_attempts"] > 0 else 0.0
    
    new_rows.append({
        "attack_condition": cond,
        "mean_runtime": row["runtime_seconds"],
        "mean_edit_distance": row["edit_distance"],
        "domain": "materials",
        "bio_cyber_label_flip_rate": np.nan,
        "bio_cyber_probability_drift": np.nan,
        "materials_Tg_absolute_drift": row["absolute_drift"],
        "materials_rdkit_proposal_validity_rate": val_rate,
        "mean_total_model_queries": row["total_model_queries"],
        "search_budget": budget,
        "mean_attack_model_queries": row["attack_model_queries"],
        "mean_attribution_queries": 0.0,
        "mean_proposal_attempts": row["proposal_attempts"]
    })
    
new_df = pd.DataFrame(new_rows)
df_final = pd.concat([df_main, new_df], ignore_index=True)
df_final = df_final.sort_values(["domain", "attack_condition", "search_budget"])
df_final.to_csv(out_path, index=False)
print("Updated cross-domain table")
