#!/usr/bin/env python3
"""Phase 6: Diagnostic Baselines."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from materials_adv.evaluation.representation_attribution import RidgeRegressor, token_features, regression_metrics

from materials_adv.data.tokenizer import tokenize
from materials_adv.evaluation.representation_attribution import token_features


def get_rdkit_features(smiles_list: list[str]) -> np.ndarray:
    feats = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            # Fallback to zeros if RDKit cannot parse (should not happen on clean data)
            feats.append([0.0]*7)
            continue
        feats.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol)
        ])
    return np.array(feats)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    with (data_dir / "vocab.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]
    
    train_reps = train_df["original_representation"].tolist()
    val_reps = val_df["original_representation"].tolist()
    
    y_train = train_df["property_value"].to_numpy()
    y_val = val_df["property_value"].to_numpy()
    
    results = []

    def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray):
        metrics = regression_metrics(y_true, y_pred)
        results.append({
            "model": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"]
        })

    # A. Target-Mean Baseline
    mean_val = np.mean(y_train)
    y_pred_mean = np.full_like(y_val, fill_value=mean_val)
    evaluate("target_mean", y_val, y_pred_mean)
    
    # B. Length-Only Regression
    train_lens = np.array([[len(tokenize(r))] for r in train_reps])
    val_lens = np.array([[len(tokenize(r))] for r in val_reps])
    
    len_model = RidgeRegressor(alpha=1.0)
    len_model.fit(train_lens, y_train)
    evaluate("length_only", y_val, len_model.predict(val_lens))
    
    # C. Token-Count Regression
    train_tc = token_features(train_reps, vocab, frequencies=False)
    val_tc = token_features(val_reps, vocab, frequencies=False)
    
    tc_model = RidgeRegressor(alpha=1.0)
    tc_model.fit(train_tc, y_train)
    evaluate("token_count", y_val, tc_model.predict(val_tc))
    
    # D. Token-Frequency Regression
    train_tf = token_features(train_reps, vocab, frequencies=True)
    val_tf = token_features(val_reps, vocab, frequencies=True)
    
    tf_model = RidgeRegressor(alpha=1.0)
    tf_model.fit(train_tf, y_train)
    evaluate("token_frequency", y_val, tf_model.predict(val_tf))
    
    # E. RDKit Descriptor Regression
    train_rdkit = get_rdkit_features(train_reps)
    val_rdkit = get_rdkit_features(val_reps)
    
    rdkit_model = RidgeRegressor(alpha=1.0)
    rdkit_model.fit(train_rdkit, y_train)
    evaluate("rdkit_descriptors", y_val, rdkit_model.predict(val_rdkit))
    
    # Save results
    out_dir = repo_root / "results" / "phase6_attribution" / "run_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    res_df.to_csv(out_dir / "simple_baseline_metrics.csv", index=False)


if __name__ == "__main__":
    main()
