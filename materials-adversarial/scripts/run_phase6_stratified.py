#!/usr/bin/env python3
"""Phase 6: Stratified Error Analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from materials_adv.data.tokenizer import tokenize
from materials_adv.evaluation.representation_attribution import regression_metrics
from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab


def get_heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return Descriptors.HeavyAtomCount(mol)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "results" / "models"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    
    vocab = load_vocab()
    
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
        
    val_df = df.iloc[splits["val"]].copy()
    val_reps = val_df["original_representation"].tolist()
    y_true = val_df["property_value"].to_numpy()
    
    val_df["seq_length"] = [len(tokenize(r)) for r in val_reps]
    val_df["heavy_atom_count"] = [get_heavy_atom_count(r) for r in val_reps]
    
    val_df["length_q"] = pd.qcut(val_df["seq_length"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    val_df["hac_q"] = pd.qcut(val_df["heavy_atom_count"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "architecture_control": load_two_branch_model(vocab, models_dir / "specialized_control"),
        "randomization_robust": load_two_branch_model(vocab, p4_dir / "rand_robust_1.0"),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    
    length_results = []
    hac_results = []
    
    for name, model in models.items():
        print(f"Predicting {name}...")
        val_df[f"{name}_pred"] = model.predict(val_reps)
        
    # Analyze by Length
    for q in val_df["length_q"].dropna().unique():
        sub_df = val_df[val_df["length_q"] == q]
        for name in models.keys():
            metrics = regression_metrics(sub_df["property_value"].to_numpy(), sub_df[f"{name}_pred"].to_numpy())
            length_results.append({
                "model": name,
                "quartile": q,
                "n_samples": len(sub_df),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"]
            })
            
    # Analyze by HAC
    for q in val_df["hac_q"].dropna().unique():
        sub_df = val_df[val_df["hac_q"] == q]
        for name in models.keys():
            metrics = regression_metrics(sub_df["property_value"].to_numpy(), sub_df[f"{name}_pred"].to_numpy())
            hac_results.append({
                "model": name,
                "quartile": q,
                "n_samples": len(sub_df),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"]
            })

    out_dir = repo_root / "results" / "phase6_attribution" / "run_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(length_results).to_csv(out_dir / "length_stratified_metrics.csv", index=False)
    pd.DataFrame(hac_results).to_csv(out_dir / "complexity_stratified_metrics.csv", index=False)
    print("Stratified analysis saved.")


if __name__ == "__main__":
    main()
