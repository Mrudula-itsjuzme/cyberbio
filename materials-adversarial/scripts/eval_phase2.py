import json
import pandas as pd
import numpy as np

def analyze_results(filepath, mae_threshold=0.4619):
    records = []
    with open(filepath, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    df = pd.DataFrame(records)
    valid_mask = df['validity_status'] == 'valid'
    success_mask = valid_mask & (df['absolute_prediction_drift'] > mae_threshold)
    
    total = len(df)
    valid = valid_mask.sum()
    successes = success_mask.sum()
    avg_drift = df.loc[valid_mask, 'absolute_prediction_drift'].mean()
    
    return {
        "total": total,
        "valid": int(valid),
        "successes": int(successes),
        "success_rate": successes / total,
        "avg_drift": float(avg_drift)
    }

def main():
    # Evaluate Baseline
    print("Evaluating Baseline Model Attacks (Test Set)...")
    baseline_stats = analyze_results("data/attacks/phase2_test_baseline.jsonl")
    print(baseline_stats)
    
    # Evaluate Defended
    print("\nEvaluating Defended Model Attacks (Test Set)...")
    defended_stats = analyze_results("data/attacks/phase2_test_defended.jsonl")
    print(defended_stats)
    
    print("\n--- Improvement ---")
    print(f"Baseline Success Rate: {baseline_stats['success_rate']:.2%}")
    print(f"Defended Success Rate: {defended_stats['success_rate']:.2%}")
    reduction = 1 - (defended_stats['success_rate'] / max(1e-9, baseline_stats['success_rate']))
    print(f"Relative Attack Success Reduction: {reduction:.1%}")

if __name__ == "__main__":
    main()
