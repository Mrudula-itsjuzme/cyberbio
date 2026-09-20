import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path("/home/mrudula/Downloads/DL_cyberbio/.phase5b-worktree/materials-adversarial")
OUT = ROOT / "results/invariance_defense"
MODELS = OUT / "models"

def run():
    from scripts.run_invariance_defense import evaluate_threat, load_inputs, load_test_banks, build_model
    
    frame, splits, vocab, scaler = load_inputs()
    repeat_bank, broad_bank = load_test_banks()
    
    # Evaluate baseline
    baseline = build_model(vocab, trainable=False)
    
    # Evaluate defended (broader_worst_seed20260815)
    defended = build_model(vocab, trainable=False)
    defended.load_state_dict(torch.load(MODELS / "broader_worst/20260815/model.pt", map_location="cpu", weights_only=True))
    
    # ordinary_augmentation (seed20260815)
    oa = build_model(vocab, trainable=False)
    oa.load_state_dict(torch.load(MODELS / "ordinary_augmentation/20260815/model.pt", map_location="cpu", weights_only=True))

    # repeat_shift_average (seed20260815)
    rsa = build_model(vocab, trainable=False)
    rsa.load_state_dict(torch.load(MODELS / "repeat_shift_average/20260815/model.pt", map_location="cpu", weights_only=True))

    # broader_average (seed20260815)
    ba = build_model(vocab, trainable=False)
    ba.load_state_dict(torch.load(MODELS / "broader_average/20260815/model.pt", map_location="cpu", weights_only=True))
    
    print("Evaluating models on broader_bank...")
    def eval_broad(m):
        summary, source = evaluate_threat(m, broad_bank, vocab, scaler, "broader")
        return source

    base_df = eval_broad(baseline)
    def_df = eval_broad(defended)
    
    merged = base_df.merge(def_df, on="source_id", suffixes=("_baseline", "_defended"))
    merged["reduction"] = merged["worst_case_baseline"] - merged["worst_case_defended"]
    merged["residual_drift"] = merged["worst_case_defended"]
    
    residuals = merged.sort_values("residual_drift", ascending=False).head(10)
    residuals = residuals.rename(columns={"worst_case_baseline": "baseline drift", "worst_case_defended": "defended drift"})
    residuals.to_csv(OUT / "residual_failures.csv", index=False)
    
    print("Top 10 residual failures saved.")
    
    # 1. Primary Effect Size
    b_mean = base_df.worst_case.mean()
    d_mean = def_df.worst_case.mean()
    reduction = b_mean - d_mean
    print("\n--- Primary Effect Size ---")
    print(f"Baseline mean: {b_mean:.4f}")
    print(f"Defended mean: {d_mean:.4f}")
    print(f"Absolute reduction: {reduction:.4f}")
    print(f"Relative reduction: {reduction / b_mean:.4f}")
    
    print(f"Median paired reduction: {merged.reduction.median():.4f}")
    print(f"P10 paired reduction: {merged.reduction.quantile(0.10):.4f}")
    print(f"P90 paired reduction: {merged.reduction.quantile(0.90):.4f}")
    
    improved = (merged.reduction > 1e-4).sum()
    worsened = (merged.reduction < -1e-4).sum()
    unchanged = len(merged) - improved - worsened
    print(f"Fraction improved: {improved / len(merged):.4f}")
    print(f"Fraction unchanged: {unchanged / len(merged):.4f}")
    print(f"Fraction worsened: {worsened / len(merged):.4f}")

    print("\n--- Direct Paired Tests ---")
    oa_df = eval_broad(oa)
    ba_df = eval_broad(ba)
    rsa_df = eval_broad(rsa)
    
    def direct_compare(name_a, df_a, name_b, df_b):
        m = df_a.merge(df_b, on="source_id", suffixes=("_a", "_b"))
        diff = m.worst_case_a - m.worst_case_b
        rng = np.random.default_rng(20260815)
        means = np.empty(5000)
        vals = diff.to_numpy()
        for i in range(5000):
            means[i] = rng.choice(vals, len(vals), replace=True).mean()
        stat, p = wilcoxon(vals)
        print(f"{name_a} vs {name_b}: mean diff={vals.mean():.4f}, median diff={np.median(vals):.4f}, CI=[{np.quantile(means,0.025):.4f}, {np.quantile(means,0.975):.4f}], W={stat}, p={p:.4e}, effect={vals.mean()/vals.std():.4f}")
    
    direct_compare("repeat_shift_average", rsa_df, "broader_average", ba_df)
    direct_compare("repeat_shift_average", rsa_df, "broader_worst", def_df)
    direct_compare("broader_average", ba_df, "broader_worst", def_df)
    direct_compare("ordinary_augmentation", oa_df, "broader_average", ba_df)

if __name__ == "__main__":
    run()
