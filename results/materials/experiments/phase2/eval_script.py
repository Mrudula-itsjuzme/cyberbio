import json
import pandas as pd
import numpy as np

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return mae, rmse, r2

def evaluate_model(attacks_file, model_name, mae_threshold=0.4619):
    records = []
    with open(attacks_file, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    df = pd.DataFrame(records)
    valid_df = df[df['validity_status'] == 'valid'].copy()
    return valid_df

def main():
    proc_df = pd.read_csv("data/processed/processed.csv")
    with open("data/processed/splits.json", "r") as f:
        splits = json.load(f)
    test_idx = splits["test"]
    
    s_to_true = {}
    for idx in test_idx:
        s_to_true[f"sample_{idx}"] = proc_df.iloc[idx]['property_value']

    def analyze(filepath):
        valid_df = evaluate_model(filepath, "model")
        valid_df['true_value'] = valid_df['sample_id'].map(s_to_true)
        
        # We need to drop any NaNs if there are mapping issues
        valid_df = valid_df.dropna(subset=['true_value'])
        
        mae, rmse, r2 = compute_metrics(valid_df['true_value'].values, valid_df['adversarial_prediction'].values)
        
        drift = valid_df['absolute_prediction_drift']
        mean_drift = drift.mean()
        median_drift = drift.median()
        max_drift = drift.max()
        
        valid_df['is_success'] = valid_df['absolute_prediction_drift'] > 0.4619
        overall_sr = valid_df['is_success'].mean()
        
        breakdown = valid_df.groupby('attack_type').agg(
            total=('attack_id', 'count'),
            success_rate=('is_success', 'mean'),
            mean_drift=('absolute_prediction_drift', 'mean')
        ).to_dict('index')
        
        return {
            "adv_mae": mae, "adv_rmse": rmse, "adv_r2": r2,
            "mean_drift": mean_drift, "median_drift": median_drift, "max_drift": max_drift,
            "success_rate": overall_sr,
            "breakdown": breakdown
        }

    base_stats = analyze("data/attacks/phase2_test_baseline.jsonl")
    def_stats = analyze("data/attacks/phase2_test_defended.jsonl")
    
    print("=== BASELINE ===")
    for k, v in base_stats.items():
        if k != 'breakdown': print(f"{k}: {v:.4f}")
    print("Breakdown:")
    for k, v in base_stats['breakdown'].items():
        print(f"  {k}: SR={v['success_rate']:.2%}, Drift={v['mean_drift']:.4f}")
        
    print("\n=== DEFENDED ===")
    for k, v in def_stats.items():
        if k != 'breakdown': print(f"{k}: {v:.4f}")
    print("Breakdown:")
    for k, v in def_stats['breakdown'].items():
        print(f"  {k}: SR={v['success_rate']:.2%}, Drift={v['mean_drift']:.4f}")

    print("\n=== IMPROVEMENTS (Defended - Baseline) ===")
    print(f"Adv MAE Error Diff: {def_stats['adv_mae'] - base_stats['adv_mae']:.4f} eV")
    print(f"Mean Drift Diff: {def_stats['mean_drift'] - base_stats['mean_drift']:.4f} eV")
    print(f"Success Rate Diff: {(def_stats['success_rate'] - base_stats['success_rate']) * 100:.2f}%")
    
if __name__ == "__main__":
    main()
