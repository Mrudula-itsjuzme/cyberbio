#!/usr/bin/env python3
"""Phase 5 Preflight Audit: Validate Phase 4 models and get canonical clean metrics."""

import hashlib
import json
from pathlib import Path
import pandas as pd

from materials_adv.evaluation.metrics import regression_metrics
from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def main():
    repo_root = Path(__file__).resolve().parent.parent
    vocab = load_vocab()
    
    models_dir = repo_root / "results" / "models"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    
    paths = {
        "ordinary_baseline": models_dir / "transformer_regressor",
        "architecture_control": models_dir / "specialized_control",
        "randomization_robust": p4_dir / "rand_robust_1.0",
        "mixed_robust": p4_dir / "mix_robust_0.1",
    }
    
    print("--- Checkpoint Hashes ---")
    for name, path in paths.items():
        if (path / "model.pt").exists():
            print(f"{name} model.pt: {sha256(path / 'model.pt')}")
        if (path / "scaler.json").exists():
            print(f"{name} scaler.json: {sha256(path / 'scaler.json')}")
            
    print("\n--- Canonical Clean Validation Metrics ---")
    
    # Load validation data
    data_dir = repo_root / "data" / "processed"
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    val_df = df.iloc[splits["val"]]
    val_reps = val_df["original_representation"].tolist()
    val_targets = val_df["property_value"].tolist()
    
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "architecture_control": load_two_branch_model(vocab, paths["architecture_control"]),
        "randomization_robust": load_two_branch_model(vocab, paths["randomization_robust"]),
        "mixed_robust": load_two_branch_model(vocab, paths["mixed_robust"]),
    }
    
    results = {}
    for name, model in models.items():
        preds = model.predict(val_reps)
        metrics = regression_metrics(val_targets, preds)
        results[name] = metrics
        print(f"{name}:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R^2: {metrics['r2']:.4f}")
        
    out_dir = repo_root / "results" / "phase5_deletion_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "preflight_audit.json").open("w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
