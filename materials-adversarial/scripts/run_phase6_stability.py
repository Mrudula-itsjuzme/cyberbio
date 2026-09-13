#!/usr/bin/env python3
"""Phase 6: Stability & Embeddings Analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab


def get_embeddings(model, reps):
    if hasattr(model, "get_branch_embeddings"):
        _, z_repr, z_chem = model.get_branch_embeddings(reps)
        # We can just return z_chem for the two branch models, as that's what drives prediction
        return z_chem
    else:
        return model.encode(reps)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "results" / "models"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    banks_dir = repo_root / "results" / "candidate_banks" / "two_branch_validation_phase3"
    
    vocab = load_vocab()
    
    from collections import defaultdict
    randomized_bank = defaultdict(list)
    with (banks_dir / "randomization_candidates.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            randomized_bank[rec["original_representation"]].append(rec["candidate_representation"])
        
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "architecture_control": load_two_branch_model(vocab, models_dir / "specialized_control"),
        "randomization_robust": load_two_branch_model(vocab, p4_dir / "rand_robust_1.0"),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    
    out_dir = repo_root / "results" / "phase6_attribution" / "run_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prediction Smoothing Check
    print("Running Prediction Smoothing Check...")
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    val_df = df.iloc[splits["val"]]
    val_reps = val_df["original_representation"].tolist()
    
    smoothing_results = []
    for name, model in models.items():
        preds = model.predict(val_reps)
        smoothing_results.append({
            "model": name,
            "pred_mean": np.mean(preds),
            "pred_std": np.std(preds),
            "pred_min": np.min(preds),
            "pred_max": np.max(preds),
            "pred_range": np.max(preds) - np.min(preds)
        })
    pd.DataFrame(smoothing_results).to_csv(out_dir / "prediction_collapse_check.csv", index=False)
    print("Smoothing check saved.")
    
    # 2. Embedding Stability & Prediction Drift
    print("Running Embedding & Prediction Stability...")
    # Use subset for speed
    subset_keys = list(randomized_bank.keys())[:100]
    
    stability_results = []
    
    for name, model in models.items():
        print(f"  {name}...")
        mean_pred_drifts = []
        mean_emb_dists = []
        
        for k in subset_keys:
            sources = [k] + randomized_bank[k]
            preds = model.predict(sources)
            pred_drift = np.max(preds) - np.min(preds)
            mean_pred_drifts.append(pred_drift)
            
            embs = get_embeddings(model, sources)
            # pairwise distances
            dists = pdist(embs, metric='euclidean')
            mean_emb_dists.append(np.mean(dists) if len(dists) > 0 else 0)
            
        stability_results.append({
            "model": name,
            "mean_max_prediction_drift": np.mean(mean_pred_drifts),
            "mean_pairwise_embedding_dist": np.mean(mean_emb_dists)
        })
        
    pd.DataFrame(stability_results).to_csv(out_dir / "embedding_stability.csv", index=False)
    print("Stability saved.")
    
if __name__ == "__main__":
    main()
