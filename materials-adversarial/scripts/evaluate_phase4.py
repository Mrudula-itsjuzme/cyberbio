#!/usr/bin/env python3
"""Evaluate Phase 4 Frozen Candidate Banks.

Evaluates:
1. Ordinary Baseline
2. Architecture Control
3. Randomization-Robust
4. Mixed Robust

Calculates:
- Primary 2x4 Robustness Matrix
- Paired model comparisons with bootstrapping (vs Baseline AND vs Architecture Control)
- Clean Performance
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from materials_adv.data.scaler import TargetScaler
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.specialized_transformer import (
    TwoBranchTransformerRegressor,
    TwoBranchTransformerRegressorModel,
)
from materials_adv.models.transformer import TransformerRegressorModel


def load_vocab() -> list[str]:
    repo_root = Path(__file__).resolve().parent.parent
    vocab_path = repo_root / "data" / "processed" / "vocab.json"
    with vocab_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ordinary_baseline(vocab: list[str]) -> TransformerRegressor:
    repo_root = Path(__file__).resolve().parent.parent
    model_dir = repo_root / "results" / "models" / "transformer_regressor"
    
    scaler = TargetScaler.load(model_dir / "scaler.json")
    
    arch = {
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "max_seq_len": 256,
        "pooling": "mean"
    }
    
    model = TransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        pooling=arch["pooling"],
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    
    return TransformerRegressor(model, vocab, scaler)


def load_two_branch_model(vocab: list[str], path: Path) -> TwoBranchTransformerRegressor:
    with (path / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
        arch = metrics["architecture"]

    scaler = TargetScaler.load(path / "scaler.json")
    
    model = TwoBranchTransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        pooling=arch["pooling"],
        branch_dim=arch.get("branch_dim", 32),
    )
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    
    return TwoBranchTransformerRegressor(model, vocab, scaler)


def bootstrap_ci(diffs: np.ndarray, source_ids: np.ndarray, n_bootstraps: int = 1000) -> tuple[float, float]:
    """Bootstrap 95% CI over SOURCE POLYMERS."""
    unique_sources = np.unique(source_ids)
    n_sources = len(unique_sources)
    
    means = []
    source_means = {s: diffs[source_ids == s].mean() for s in unique_sources}
    for _ in range(n_bootstraps):
        sampled_sources = np.random.choice(unique_sources, size=n_sources, replace=True)
        sample_mean = np.mean([source_means[s] for s in sampled_sources])
        means.append(sample_mean)
        
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(run_id: str, rand_dir: str, mix_dir: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    banks_dir = repo_root / "results" / "candidate_banks" / "two_branch_validation_phase3"
    models_dir = repo_root / "results" / "models"
    out_dir = repo_root / "results" / "phase4_controlled_robustness" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = load_vocab()
    
    m_base = load_ordinary_baseline(vocab)
    m_ctrl = load_two_branch_model(vocab, models_dir / "specialized_control")
    
    models = {
        "ordinary_baseline": m_base,
        "architecture_control": m_ctrl,
    }
    
    if rand_dir:
        p4_rand_dir = repo_root / "results" / "phase4_controlled_robustness" / rand_dir
        m_rand = load_two_branch_model(vocab, p4_rand_dir)
        models["randomization_robust"] = m_rand

    if mix_dir:
        p4_mix_dir = repo_root / "results" / "phase4_controlled_robustness" / mix_dir
        m_mix = load_two_branch_model(vocab, p4_mix_dir)
        models["mixed_robust"] = m_mix


    # Load candidates
    rand_path = banks_dir / "randomization_candidates.jsonl"
    sub_path = banks_dir / "substitution_candidates.jsonl"
    
    with rand_path.open("r", encoding="utf-8") as f:
        rand_recs = [json.loads(line) for line in f]
    with sub_path.open("r", encoding="utf-8") as f:
        sub_recs = [json.loads(line) for line in f]
        
    df_rand = pd.DataFrame(rand_recs)
    df_sub = pd.DataFrame(sub_recs)
    
    df_rand = df_rand[df_rand["valid"] == True]
    df_sub = df_sub[df_sub["valid"] == True]
    
    # Baseline Clean MAE (for stress exceedance)
    unique_sources_df = df_rand.drop_duplicates("source_id")
    clean_reps = unique_sources_df["original_representation"].tolist()
    clean_targets = unique_sources_df["target"].values
    
    baseline_clean_preds = m_base.predict(clean_reps)
    baseline_clean_mae = float(np.mean(np.abs(baseline_clean_preds - clean_targets)))

    robustness_matrix = []
    paired_comparisons = []
    
    for df, attack_name in [(df_rand, "randomization"), (df_sub, "substitution")]:
        sources = df["source_id"].values
        orig_reps = df["original_representation"].tolist()
        cand_reps = df["candidate_representation"].tolist()
        
        preds = {}
        for name, m in models.items():
            if name == "ordinary_baseline":
                p_clean = np.array(m.predict(orig_reps))
                p_cand = np.array(m.predict(cand_reps))
                preds[name] = (p_clean, p_cand)
            else:
                p_clean, _, _ = m.get_branch_embeddings(orig_reps)
                p_cand, _, _ = m.get_branch_embeddings(cand_reps)
                preds[name] = (p_clean, p_cand)
                
        # Metric calculation per model
        for name in models.keys():
            p_clean, p_cand = preds[name]
            abs_drift = np.abs(p_clean - p_cand)
            
            row = {
                "model": name,
                "attack": attack_name,
                "source_n": len(np.unique(sources)),
                "candidate_n": len(df),
                "mean_drift": float(np.mean(abs_drift)),
                "median_drift": float(np.median(abs_drift)),
                "p90_drift": float(np.percentile(abs_drift, 90)),
                "p95_drift": float(np.percentile(abs_drift, 95)),
                "max_drift": float(np.max(abs_drift)),
            }
            
            if attack_name == "randomization":
                df_temp = pd.DataFrame({"source": sources, "p_cand": p_cand})
                grouped = df_temp.groupby("source")["p_cand"]
                
                row["mean_source_std"] = float(grouped.std().mean())
                row["mean_source_range"] = float((grouped.max() - grouped.min()).mean())
                row["invariance_score"] = float(max(0.0, 1.0 - (row["mean_drift"] / baseline_clean_mae)))
            else:
                exceedance = np.mean(abs_drift > baseline_clean_mae)
                row["stress_exceedance_rate"] = float(exceedance)
                
            robustness_matrix.append(row)
            
            # Paired comparisons
            for ref_name in ["ordinary_baseline", "architecture_control"]:
                if name != ref_name:
                    ref_drift = np.abs(preds[ref_name][0] - preds[ref_name][1])
                    improvement = ref_drift - abs_drift
                    
                    ci_low, ci_high = bootstrap_ci(improvement, sources)
                    
                    paired_comparisons.append({
                        "model": name,
                        "reference_model": ref_name,
                        "attack": attack_name,
                        "paired_n": len(df),
                        "mean_improvement": float(np.mean(improvement)),
                        "median_improvement": float(np.median(improvement)),
                        "ci_95_low": ci_low,
                        "ci_95_high": ci_high,
                        "frac_improved": float(np.mean(improvement > 0)),
                        "frac_worsened": float(np.mean(improvement < 0)),
                    })
                    
    # Save Outputs
    pd.DataFrame(robustness_matrix).to_csv(out_dir / "robustness_matrix.csv", index=False)
    pd.DataFrame(paired_comparisons).to_csv(out_dir / "bootstrap_results.csv", index=False)
    
    # Collect clean metrics from their respective metrics.json
    clean_metrics = []
    
    # Base paths
    metrics_paths = [
        ("ordinary_baseline", models_dir / "transformer_regressor"),
        ("architecture_control", models_dir / "specialized_control"),
    ]
    if rand_dir:
        metrics_paths.append(("randomization_robust", repo_root / "results" / "phase4_controlled_robustness" / rand_dir))
    if mix_dir:
        metrics_paths.append(("mixed_robust", repo_root / "results" / "phase4_controlled_robustness" / mix_dir))

    for name, path in metrics_paths:
        with (path / "metrics.json").open("r", encoding="utf-8") as f:
            m = json.load(f)
            clean_metrics.append({
                "model": name,
                "val_mae": m.get("best_val_mae", None),
                "val_rmse": m.get("val_rmse", None),
                "val_r2": m.get("val_r2", None),
                "test_mae_exposed_ref": m.get("exposed_test_reference", {}).get("test_mae", m.get("test_mae", None)),
                "test_rmse_exposed_ref": m.get("exposed_test_reference", {}).get("test_rmse", m.get("test_rmse", None)),
                "test_r2_exposed_ref": m.get("exposed_test_reference", {}).get("test_r2", m.get("test_r2", None)),
            })
            
    pd.DataFrame(clean_metrics).to_csv(out_dir / "clean_metrics.csv", index=False)
    
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "baseline_clean_mae": baseline_clean_mae,
        }, f, indent=2)

    print(f"Phase 4 Evaluation Completed. Outputs saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--rand-dir", type=str, default="")
    parser.add_argument("--mix-dir", type=str, default="")
    args = parser.parse_args()
    evaluate(args.run_id, args.rand_dir, args.mix_dir)
