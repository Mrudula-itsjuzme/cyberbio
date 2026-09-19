import pandas as pd
import numpy as np
import os
import sys

# Set up paths
sys.path.append(os.path.abspath("src"))
from materials_adv.data.tokenizer import tokenize
from scripts.evaluate_phase4 import load_ordinary_baseline, load_vocab

print("Loading vocabulary...")
vocab = load_vocab()
print("Loading model...")
model = load_ordinary_baseline(vocab)
print("Model loaded.")

# Load candidates from comparison-experiments
print("Loading MCMC candidates...")
df = pd.read_csv("../comparison-experiments/results/summaries/budget_sweep_results.csv")
# Filter for MCMC, budget 50
df_mcmc = df[(df['attack_family'] == 'mcmc') & (df['query_budget'] == 50)].copy()
# Pick 20 unique sources
source_ids = df_mcmc['source_id'].unique()[:20]

results = []

for sid in source_ids:
    subset = df_mcmc[df_mcmc['source_id'] == sid]
    if len(subset) == 0: continue
    row = subset.iloc[0]
    
    source_seq = str(row['source_sequence'])
    adv_seq = str(row['candidate_sequence'])
    
    # 1. Base prediction
    base_tokens = tokenize(source_seq)
    base_pred = model.predict([source_seq])[0]
    
    for i in range(len(base_tokens)):
        occ_tokens = base_tokens[:i] + base_tokens[i+1:]
        if not occ_tokens: continue
        occ_rep = "".join(occ_tokens)
        try:
            occ_pred = model.predict([occ_rep])[0]
            delta = abs(base_pred - occ_pred)
            results.append({
                "source_id": sid,
                "type": "clean",
                "token": base_tokens[i],
                "position": i,
                "delta": float(delta)
            })
        except Exception:
            pass

    # 2. Adversarial prediction
    adv_tokens = tokenize(adv_seq)
    adv_pred = model.predict([adv_seq])[0]
    
    for i in range(len(adv_tokens)):
        occ_tokens = adv_tokens[:i] + adv_tokens[i+1:]
        if not occ_tokens: continue
        occ_rep = "".join(occ_tokens)
        try:
            occ_pred = model.predict([occ_rep])[0]
            delta = abs(adv_pred - occ_pred)
            results.append({
                "source_id": sid,
                "type": "mcmc_adv",
                "token": adv_tokens[i],
                "position": i,
                "delta": float(delta)
            })
        except Exception:
            pass

out_dir = "../comparison-experiments/results/raw_explainability"
os.makedirs(out_dir, exist_ok=True)
res_df = pd.DataFrame(results)
res_df.to_csv(out_dir + "/mcmc_occlusion.csv", index=False)
print(f"Finished explainability experiment for {len(source_ids)} sources.")
