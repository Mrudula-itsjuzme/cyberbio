#!/usr/bin/env python3
"""Evaluate Phase 5 Deletion Transfer.

Evaluates:
1. Ordinary Baseline
2. Architecture Control
3. Randomization-Robust (Phase 4)
4. Mixed Robust (Phase 4)

On the newly generated Unseen Deletion bank.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab, bootstrap_ci

FIXED_STRESS_THRESHOLD = 0.4862  # Clean ordinary_baseline validation MAE

def evaluate_deletion() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    banks_dir = repo_root / "results" / "candidate_banks" / "deletion_transfer_phase5"
    models_dir = repo_root / "results" / "models"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    out_dir = repo_root / "results" / "phase5_deletion_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = load_vocab()
    
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "architecture_control": load_two_branch_model(vocab, models_dir / "specialized_control"),
        "randomization_robust": load_two_branch_model(vocab, p4_dir / "rand_robust_1.0"),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    
    # Load candidates
    del_path = banks_dir / "deletion_candidates.jsonl"
    with del_path.open("r", encoding="utf-8") as f:
        del_recs = [json.loads(line) for line in f]
        
    df = pd.DataFrame(del_recs)
    
    source_reps = df["original_representation"].tolist()
    adv_reps = df["candidate_representation"].tolist()
    source_ids = df["source_id"].to_numpy()
    
    print(f"Evaluating {len(adv_reps)} deletion candidates...")
    
    results = {}
    drifts = {}
    
    for name, model in models.items():
        print(f"  Predicting with {name}...")
        preds_clean = np.array(model.predict(source_reps))
        preds_adv = np.array(model.predict(adv_reps))
        
        abs_drift = np.abs(preds_adv - preds_clean)
        drifts[name] = abs_drift
        
        results[name] = {
            "mean_drift": float(np.mean(abs_drift)),
            "median_drift": float(np.median(abs_drift)),
            "p90_drift": float(np.percentile(abs_drift, 90)),
            "p95_drift": float(np.percentile(abs_drift, 95)),
            "max_drift": float(np.max(abs_drift)),
            "stress_exceedance_rate": float(np.mean(abs_drift > FIXED_STRESS_THRESHOLD)),
            "fixed_stress_threshold": FIXED_STRESS_THRESHOLD
        }

    # Paired comparisons
    comparisons = []
    
    pairs = [
        ("ordinary_baseline", "architecture_control"),
        ("ordinary_baseline", "randomization_robust"),
        ("ordinary_baseline", "mixed_robust"),
        ("architecture_control", "randomization_robust"),
        ("architecture_control", "mixed_robust"),
        ("randomization_robust", "mixed_robust"),
    ]
    
    for ref_name, hyp_name in pairs:
        ref_drift = drifts[ref_name]
        hyp_drift = drifts[hyp_name]
        
        diff = hyp_drift - ref_drift
        
        ci_lower, ci_upper = bootstrap_ci(diff, source_ids)
        mean_diff = float(np.mean(diff))
        
        significant = (ci_upper < 0) or (ci_lower > 0)
        direction = "improved" if ci_upper < 0 else ("worsened" if ci_lower > 0 else "null")
        
        comparisons.append({
            "reference": ref_name,
            "hypothesis": hyp_name,
            "metric": "absolute_drift",
            "mean_difference": mean_diff,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "significant": significant,
            "direction": direction
        })

    report = {
        "models": results,
        "paired_generalization": comparisons,
        "n_candidates": len(df),
        "n_source_polymers": len(np.unique(source_ids))
    }
    
    with (out_dir / "deletion_evaluation.json").open("w") as f:
        json.dump(report, f, indent=2)
        
    print("\n--- RESULTS ---")
    for name, m in results.items():
        print(f"\n{name.upper()}")
        print(f"  Mean Drift: {m['mean_drift']:.4f}")
        print(f"  Stress Exceedance: {m['stress_exceedance_rate']:.1%}")
        
    print("\n--- PAIRED IMPROVEMENTS ---")
    for comp in comparisons:
        if comp["significant"]:
            print(f"{comp['hypothesis']} vs {comp['reference']}: {comp['direction'].upper()} (Diff: {comp['mean_difference']:.4f}, CI: [{comp['ci_95_lower']:.4f}, {comp['ci_95_upper']:.4f}])")
        else:
            print(f"{comp['hypothesis']} vs {comp['reference']}: NULL")


if __name__ == "__main__":
    evaluate_deletion()
