#!/usr/bin/env python3
"""Phase 6: Token Occlusion Analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from materials_adv.data.tokenizer import tokenize
from materials_adv.evaluation.representation_attribution import occlusion_sensitivity
from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab


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
    # Use deterministic subset of first 100 validation polymers
    subset_reps = val_df["original_representation"].tolist()[:100]
    
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "architecture_control": load_two_branch_model(vocab, models_dir / "specialized_control"),
        "randomization_robust": load_two_branch_model(vocab, p4_dir / "rand_robust_1.0"),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    
    results = []
    
    # Run occlusion for all 100 polymers for all 4 models
    # Note: occlusion_sensitivity deletes one token at a time and measures prediction change
    for name, model in models.items():
        print(f"Running occlusion for {name}...")
        
        for rep in subset_reps:
            tokens = tokenize(rep)
            base_pred = model.predict([rep])[0]
            
            # Predict missing one token at a time
            for i in range(len(tokens)):
                occluded_tokens = tokens[:i] + tokens[i+1:]
                if not occluded_tokens:
                    continue
                occluded_rep = "".join(occluded_tokens)
                try:
                    occ_pred = model.predict([occluded_rep])[0]
                    delta = abs(base_pred - occ_pred)
                    
                    results.append({
                        "model": name,
                        "token": tokens[i],
                        "position": i,
                        "relative_position": i / len(tokens),
                        "seq_length": len(tokens),
                        "delta": float(delta)
                    })
                except Exception:
                    pass
                    
    out_dir = repo_root / "results" / "phase6_attribution" / "run_01"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "token_occlusion.csv", index=False)
    
    # Print summary by token type and model
    summary = res_df.groupby(["model", "token"])["delta"].mean().unstack(level=0)
    print("Mean Occlusion Sensitivity by Token:")
    print(summary.head(20)) # Show top 20 tokens


if __name__ == "__main__":
    main()
