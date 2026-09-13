#!/usr/bin/env python3
"""Run Phase 7 Attack Transfer Evaluation.

Evaluates candidates generated against one model on the other model.
"""

import json
from pathlib import Path

import numpy as np

from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab

def run_transfer():
    repo_root = Path(__file__).resolve().parent.parent
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    search_dir = repo_root / "results" / "phase7_adaptive_search"
    search_results_path = search_dir / "search_trajectories.jsonl"
    out_file = search_dir / "transfer_evaluation.json"
    
    if not search_results_path.exists():
        print(f"File not found: {search_results_path}")
        return
        
    vocab = load_vocab()
    
    print("Loading models...")
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    
    # Load all budget=50 results
    records = []
    with search_results_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["budget"] == 50:
                records.append(rec)
                
    if not records:
        print("No budget=50 records found.")
        return
        
    print(f"Evaluating transfer on {len(records)} candidates from budget 50...")
    
    transfer_results = []
    
    # Predict with all models
    for rec in records:
        target_model = rec["model"]
        source_id = rec["source_id"]
        original = rec["original_representation"]
        adv = rec["best_representation"]
        strategy = rec["strategy"]
        
        eval_models = [m for m in models.keys() if m != target_model]
        
        for eval_model_name in eval_models:
            model = models[eval_model_name]
            
            clean_pred = float(model.predict([original])[0])
            adv_pred = float(model.predict([adv])[0])
            
            drift = abs(adv_pred - clean_pred)
            
            transfer_results.append({
                "source_model": target_model,
                "eval_model": eval_model_name,
                "strategy": strategy,
                "source_id": source_id,
                "transfer_drift": drift,
                "clean_prediction": clean_pred,
                "adv_prediction": adv_pred,
                "target_drift": rec["best_drift"]
            })
            
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(transfer_results, f, indent=2)
        
    print(f"Transfer evaluation saved to {out_file}")

if __name__ == "__main__":
    run_transfer()
