import json
import yaml
import pandas as pd
from pathlib import Path

with open("configs/dataset.yaml") as f:
    cfg = yaml.safe_load(f)

rep_col = cfg["representation_column"]
tgt_col = cfg["target_column"]

df = pd.read_csv("data/processed/processed.csv")
with open("data/processed/splits.json") as f:
    splits = json.load(f)
train_df = df.iloc[splits["train"]].copy()

with open("data/attacks/phase3_residualized_attacks.jsonl") as f:
    records = [json.loads(line) for line in f]
    
mcmc_attacks = [r for r in records if r["attack_type"] == "probabilistic_mcmc" and r["validity_status"] == "valid"]

target_map = {r[rep_col]: r[tgt_col] for _, r in train_df.iterrows()}

new_rows = []
for r in mcmc_attacks:
    orig_psmiles = r["original_psmiles"]
    if orig_psmiles in target_map:
        new_rows.append({
            rep_col: r["adversarial_psmiles"],
            tgt_col: target_map[orig_psmiles],
            "dataset": "mcmc_augmented"
        })

if new_rows:
    aug_df = pd.DataFrame(new_rows)
    combined = pd.concat([train_df, aug_df], ignore_index=True)
    combined.to_csv("data/processed/train_mcmc_augmented.csv", index=False)
    print(f"Created augmented dataset with {len(new_rows)} MCMC examples.")
else:
    print("No valid MCMC examples found for training set.")
