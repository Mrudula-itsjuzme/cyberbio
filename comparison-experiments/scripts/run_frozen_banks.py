import pandas as pd
import json
import os
import hashlib

df = pd.read_csv("comparison-experiments/results/recomputed_metrics/classical_candidate_metrics.csv")
df_b50 = df[df['query_budget'] == 50]

out_dir = "comparison-experiments/results/frozen_candidate_banks"
os.makedirs(out_dir, exist_ok=True)

banks = {}
for attack in ["random", "mcmc", "evolutionary"]:
    subset = df_b50[df_b50['attack_family'] == attack]
    bank = []
    for _, row in subset.iterrows():
        bank.append({
            "source_id": row["source_id"],
            "source_sequence": row["source_sequence"],
            "candidate_sequence": row["candidate_sequence"],
            "seed": row["seed"],
            "budget": row["query_budget"],
            "source_prediction_original": row["source_prediction"],
            "candidate_prediction_original": row["candidate_prediction"],
            "prediction_drift_original": row["prediction_drift"],
            "constraint_pass": row["constraint_pass"],
            "Tanimoto": row["recomputed_tanimoto"],
            "edit_distance": row["recomputed_edit_distance"]
        })
    banks[attack] = bank
    
    path = f"{out_dir}/{attack}.jsonl"
    with open(path, "w") as f:
        for b in bank:
            f.write(json.dumps(b) + "\n")
            
    # Hash
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    print(f"Bank {attack} created: {len(bank)} records, hash: {h.hexdigest()[:8]}")

print("Phase D Complete.")
