import pandas as pd
import numpy as np
import json
import os
import glob
import hashlib

df = pd.read_csv("data/v3/dataset.csv")
attacks = pd.read_csv("results/v3/attack_results.csv")

# join true label
attacks = attacks.merge(df[["id", "label"]], left_on="source_id", right_on="id")
attacks = attacks.rename(columns={"label": "true_label"})
attacks["clean_correct"] = ((attacks["source_prediction"] > 0.5) == attacks["true_label"])
attacks["constraint_pass"] = True # No constraints on arbitrary DNA strings

summary_rows = []
for (attack, budget), grp in attacks.groupby(["attack", "budget"]):
    # success per query: flips / sum(queries)
    n = len(grp)
    summary_rows.append({
        "attack": attack,
        "budget": budget,
        "n": n,
        "flip_rate": grp["label_flip"].mean(),
        "mean_absolute_probability_change": grp["absolute_prediction_change"].mean(),
        "median_absolute_probability_change": grp["absolute_prediction_change"].median(),
        "p90_absolute_probability_change": grp["absolute_prediction_change"].quantile(0.9),
        "mean_edits": grp["number_of_edits"].mean(),
        "median_edits": grp["number_of_edits"].median(),
        "mean_queries": grp["query_count"].mean(),
        "success_per_query": grp["label_flip"].sum() / grp["query_count"].sum(),
        "constraint_pass_rate": grp["constraint_pass"].mean(),
        "true_class_1_flip_rate": grp[grp["true_label"]==1]["label_flip"].mean() if len(grp[grp["true_label"]==1])>0 else np.nan,
        "clean_correct_flip_rate": grp[grp["clean_correct"]==True]["label_flip"].mean() if len(grp[grp["clean_correct"]==True])>0 else np.nan
    })

pd.DataFrame(summary_rows).to_csv("results/v3/attack_summary.csv", index=False)

# Phase 5
manifest = []
for file in glob.glob("results/v3/frozen_attack_banks/*.jsonl"):
    with open(file, "rb") as f:
        data = f.read()
        sha = hashlib.sha256(data).hexdigest()
    
    records = []
    with open(file, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    manifest.append({
        "bank_file": os.path.basename(file),
        "row_count": len(records),
        "unique_source_count": len(set(r["source_id"] for r in records)),
        "duplicate_candidates": len(records) - len(set(r["candidate_sequence"] for r in records)),
        "budget": records[0]["budget"] if records else None,
        "seed_distribution": "deterministic",
        "SHA256": sha
    })
    
with open("results/v3/frozen_attack_banks/manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)

print("Phase 4 & 5 done.")
