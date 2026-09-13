#!/usr/bin/env python3
"""Evaluate Phase 3 Frozen Candidate Banks.

Evaluates the Ordinary Baseline, Architecture Control, and Specialized Model
against the frozen randomization and substitution candidate banks.

Calculates:
- Prediction robustness metrics
- Paired model comparisons with bootstrapping
- Branch-level evaluation (selectivity ratios)
- Representation collapse check (on clean validation data)
- Head/Branch ablation
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
    
    with (model_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    scaler = TargetScaler.load(model_dir / "scaler.json")
    
    # Hardcode baseline architecture from Phase 1 / 2 documentation
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
    """Bootstrap 95% CI over SOURCE POLYMERS, not individual candidates."""
    unique_sources = np.unique(source_ids)
    n_sources = len(unique_sources)
    
    means = []
    for _ in range(n_bootstraps):
        sampled_sources = np.random.choice(unique_sources, size=n_sources, replace=True)
        # For each sampled source, get its mean diff, then average those
        # Or more simply: just average the diffs of candidates belonging to sampled sources.
        # But properly, we should resample sources and take all their candidates.
        
        # Fast way: precompute mean diff per source
        source_means = {s: diffs[source_ids == s].mean() for s in unique_sources}
        sample_mean = np.mean([source_means[s] for s in sampled_sources])
        means.append(sample_mean)
        
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def get_ablation_metrics(
    model_wrapper: TwoBranchTransformerRegressor,
    df_clean: pd.DataFrame,
    df_rand: pd.DataFrame,
    df_sub: pd.DataFrame,
    ablate_branch: str,
) -> dict[str, float]:
    """Zero out branch A or B and compute clean MAE and drifts."""
    model = model_wrapper.model
    scaler = model_wrapper.scaler
    
    def predict_ablated(reps: list[str]) -> np.ndarray:
        src, mask = model_wrapper._tokenize_batch(reps)
        with torch.no_grad():
            h = model.encode_shared(src, padding_mask=mask)
            z_a = model.branch_a(h)
            z_b = model.branch_b(h)
            
            if ablate_branch == "A":
                z_a = torch.zeros_like(z_a)
            elif ablate_branch == "B":
                z_b = torch.zeros_like(z_b)
                
            fusion = torch.cat([z_a, z_b], dim=-1)
            norm_preds = model.regressor(fusion).squeeze(-1).cpu().numpy()
            
        return scaler.inverse_transform(norm_preds)

    # Clean MAE
    preds_clean = predict_ablated(df_clean["original_representation"].tolist())
    targets = df_clean["target"].values
    mae_clean = float(np.mean(np.abs(preds_clean - targets)))

    # Rand Drift
    preds_rand_clean = predict_ablated(df_rand["original_representation"].tolist())
    preds_rand_cand = predict_ablated(df_rand["candidate_representation"].tolist())
    drift_rand = float(np.mean(np.abs(preds_rand_clean - preds_rand_cand)))
    
    # Sub Drift
    preds_sub_clean = predict_ablated(df_sub["original_representation"].tolist())
    preds_sub_cand = predict_ablated(df_sub["candidate_representation"].tolist())
    drift_sub = float(np.mean(np.abs(preds_sub_clean - preds_sub_cand)))

    return {
        "mae_clean": mae_clean,
        "mean_drift_rand": drift_rand,
        "mean_drift_sub": drift_sub,
    }


def evaluate(run_id: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    banks_dir = repo_root / "results" / "candidate_banks" / f"two_branch_validation_{run_id}"
    models_dir = repo_root / "results" / "models"
    
    vocab = load_vocab()
    
    m_base = load_ordinary_baseline(vocab)
    m_ctrl = load_two_branch_model(vocab, models_dir / "specialized_control")
    m_spec = load_two_branch_model(vocab, models_dir / "specialized_model")
    
    models = {
        "ordinary_baseline": m_base,
        "architecture_control": m_ctrl,
        "specialized_model": m_spec
    }

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
    # We get clean reps from df_rand, unique by source
    unique_sources_df = df_rand.drop_duplicates("source_id")
    clean_reps = unique_sources_df["original_representation"].tolist()
    clean_targets = unique_sources_df["target"].values
    
    baseline_clean_preds = m_base.predict(clean_reps)
    baseline_clean_mae = float(np.mean(np.abs(baseline_clean_preds - clean_targets)))

    robustness_matrix = []
    paired_comparisons = []
    
    # Precompute all predictions and branch vectors
    # We will compute them strictly in the same order
    for df, attack_name in [(df_rand, "randomization"), (df_sub, "substitution")]:
        sources = df["source_id"].values
        orig_reps = df["original_representation"].tolist()
        cand_reps = df["candidate_representation"].tolist()
        
        preds = {}
        for name, m in models.items():
            if name == "ordinary_baseline":
                p_clean = np.array(m.predict(orig_reps))
                p_cand = np.array(m.predict(cand_reps))
                preds[name] = (p_clean, p_cand, None, None, None, None)
            else:
                p_clean, za_clean, zb_clean = m.get_branch_embeddings(orig_reps)
                p_cand, za_cand, zb_cand = m.get_branch_embeddings(cand_reps)
                preds[name] = (p_clean, p_cand, za_clean, zb_clean, za_cand, zb_cand)
                
        # Metric calculation per model
        for name in models.keys():
            p_clean, p_cand, za_c, zb_c, za_k, zb_k = preds[name]
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
                # Per-source metrics
                df_temp = pd.DataFrame({"source": sources, "p_cand": p_cand})
                grouped = df_temp.groupby("source")["p_cand"]
                
                row["mean_source_std"] = float(grouped.std().mean())
                row["mean_source_range"] = float((grouped.max() - grouped.min()).mean())
                
                # Invariance Score (1.0 - normalized drift relative to baseline MAE)
                row["invariance_score"] = float(max(0.0, 1.0 - (row["mean_drift"] / baseline_clean_mae)))
            else:
                # Substitution: stress exceedance rate
                exceedance = np.mean(abs_drift > baseline_clean_mae)
                row["stress_exceedance_rate"] = float(exceedance)
                
            robustness_matrix.append(row)
            
            # Paired comparisons against ordinary baseline
            if name != "ordinary_baseline":
                base_drift = np.abs(preds["ordinary_baseline"][0] - preds["ordinary_baseline"][1])
                improvement = base_drift - abs_drift
                
                ci_low, ci_high = bootstrap_ci(improvement, sources)
                
                paired_comparisons.append({
                    "model": name,
                    "attack": attack_name,
                    "paired_n": len(df),
                    "mean_improvement": float(np.mean(improvement)),
                    "median_improvement": float(np.median(improvement)),
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "frac_improved": float(np.mean(improvement > 0)),
                    "frac_worsened": float(np.mean(improvement < 0)),
                })
                
    # Branch Selectivity
    branch_selectivity = []
    np.random.seed(20261101)  # For reproducibility in paired selection
    
    for name in ["architecture_control", "specialized_model"]:
        m = models[name]
        
        # Rand
        _, za_c_r, zb_c_r = m.get_branch_embeddings(df_rand["original_representation"].tolist())
        _, za_k_r, zb_k_r = m.get_branch_embeddings(df_rand["candidate_representation"].tolist())
        d_A_rand = np.linalg.norm(za_c_r - za_k_r, axis=1).mean()
        d_B_rand = np.linalg.norm(zb_c_r - zb_k_r, axis=1).mean()
        
        # Sub
        _, za_c_s, zb_c_s = m.get_branch_embeddings(df_sub["original_representation"].tolist())
        _, za_k_s, zb_k_s = m.get_branch_embeddings(df_sub["candidate_representation"].tolist())
        d_A_sub = np.linalg.norm(za_c_s - za_k_s, axis=1).mean()
        d_B_sub = np.linalg.norm(zb_c_s - zb_k_s, axis=1).mean()
        
        S_A = float(d_A_sub / d_A_rand) if d_A_rand > 0 else 0.0
        S_B = float(d_B_sub / d_B_rand) if d_B_rand > 0 else 0.0
        
        branch_selectivity.append({
            "model": name,
            "d_A_rand": float(d_A_rand),
            "d_B_rand": float(d_B_rand),
            "d_A_sub": float(d_A_sub),
            "d_B_sub": float(d_B_sub),
            "S_A": S_A,
            "S_B": S_B,
        })
        
    # Representation collapse (clean only)
    collapse_diagnostics = {}
    for name in ["architecture_control", "specialized_model"]:
        m = models[name]
        _, za, zb = m.get_branch_embeddings(clean_reps)
        
        collapse_diagnostics[name] = {
            "branch_A_var": float(np.var(za, axis=0).mean()),
            "branch_B_var": float(np.var(zb, axis=0).mean()),
            "branch_A_norm": float(np.linalg.norm(za, axis=1).mean()),
            "branch_B_norm": float(np.linalg.norm(zb, axis=1).mean()),
        }
        
    # Ablation
    ablation = {}
    for name in ["architecture_control", "specialized_model"]:
        m = models[name]
        ab_A = get_ablation_metrics(m, unique_sources_df, df_rand, df_sub, "A")
        ab_B = get_ablation_metrics(m, unique_sources_df, df_rand, df_sub, "B")
        ablation[name] = {
            "ablate_A": ab_A,
            "ablate_B": ab_B,
        }
        
    # Save Outputs
    pd.DataFrame(robustness_matrix).to_csv(banks_dir / "robustness_matrix.csv", index=False)
    pd.DataFrame(branch_selectivity).to_csv(banks_dir / "branch_selectivity.csv", index=False)
    
    with (banks_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "baseline_clean_mae": baseline_clean_mae,
            "paired_comparisons": paired_comparisons,
            "collapse_diagnostics": collapse_diagnostics,
            "ablation": ablation,
        }, f, indent=2)

    print(f"Phase 3 Evaluation Completed. Outputs saved to {banks_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.run_id)
