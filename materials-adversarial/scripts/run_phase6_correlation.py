#!/usr/bin/env python3
"""Phase 6: Correlation Audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from materials_adv.data.tokenizer import tokenize
from materials_adv.evaluation.representation_attribution import token_features
from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab


def safe_corr(x, y, func):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    res = func(x, y)
    return float(res[0])


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "results" / "models"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    
    vocab = load_vocab()
    
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
        
    val_df = df.iloc[splits["val"]]
    val_reps = val_df["original_representation"].tolist()
    y_true = val_df["property_value"].to_numpy()
    
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "architecture_control": load_two_branch_model(vocab, models_dir / "specialized_control"),
        "randomization_robust": load_two_branch_model(vocab, p4_dir / "rand_robust_1.0"),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    
    # Compute token statistics for val_reps
    lengths = np.array([len(tokenize(r)) for r in val_reps], dtype=float)
    unique_tokens = np.array([len(set(tokenize(r))) for r in val_reps], dtype=float)
    
    # Ring token counts (assume digits 1-9 are rings)
    ring_counts = np.array([sum(1 for t in tokenize(r) if t in "123456789") for r in val_reps], dtype=float)
    aromatic_counts = np.array([sum(1 for t in tokenize(r) if t.islower() and t in "cnosp") for r in val_reps], dtype=float)
    
    features = {
        "length": lengths,
        "unique_tokens": unique_tokens,
        "ring_syntax_count": ring_counts,
        "aromatic_atom_count": aromatic_counts
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Predicting with {name}...")
        y_pred = np.array(model.predict(val_reps))
        abs_err = np.abs(y_true - y_pred)
        
        for feat_name, feat_vals in features.items():
            results.append({
                "model": name,
                "feature": feat_name,
                "target": "true_bandgap",
                "pearson": safe_corr(feat_vals, y_true, pearsonr),
                "spearman": safe_corr(feat_vals, y_true, spearmanr)
            })
            results.append({
                "model": name,
                "feature": feat_name,
                "target": "prediction",
                "pearson": safe_corr(feat_vals, y_pred, pearsonr),
                "spearman": safe_corr(feat_vals, y_pred, spearmanr)
            })
            results.append({
                "model": name,
                "feature": feat_name,
                "target": "abs_error",
                "pearson": safe_corr(feat_vals, abs_err, pearsonr),
                "spearman": safe_corr(feat_vals, abs_err, spearmanr)
            })

    out_dir = repo_root / "results" / "phase6_attribution" / "run_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv(out_dir / "correlation_analysis.csv", index=False)


if __name__ == "__main__":
    main()
