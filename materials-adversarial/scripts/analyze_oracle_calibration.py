import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

def main():
    parser = argparse.ArgumentParser(description="Analyze Oracle Calibration")
    parser.add_argument("--ingested-results", type=str, required=True, help="Path to ingested_results.json")
    args = parser.parse_args()
    
    results_path = Path(args.ingested_results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)
        
    with open(results_path) as f:
        jobs = json.load(f)
        
    total_jobs = len(jobs)
    if total_jobs == 0:
        print("No jobs found in ingested results.")
        return

    converged_jobs = [j for j in jobs if j.get("converged", False)]
    failed_jobs = total_jobs - len(converged_jobs)
    convergence_rate = len(converged_jobs) / total_jobs
    
    print("=== ORACLE CALIBRATION ANALYSIS ===")
    print(f"Total jobs: {total_jobs}")
    print(f"Converged: {len(converged_jobs)} ({convergence_rate*100:.1f}%)")
    print(f"Failed: {failed_jobs}")
    
    # We don't hardcode an arbitrary acceptance threshold
    
    y_true = []
    y_pred = []
    
    per_sample = []
    
    for j in jobs:
        dataset_bg = j.get("dataset_bandgap_eV")
        oracle_bg = j.get("bandgap_eV")
        residual = None
        
        if j.get("converged", False) and dataset_bg is not None and oracle_bg is not None:
            y_true.append(dataset_bg)
            y_pred.append(oracle_bg)
            residual = oracle_bg - dataset_bg
            
        per_sample.append({
            "job_id": j.get("job_id"),
            "dataset_bandgap": dataset_bg,
            "oracle_bandgap": oracle_bg,
            "residual": residual,
            "convergence": j.get("converged", False),
            "warnings": "; ".join(j.get("warnings", []))
        })
            
    # Persist per-sample results
    ps_df = pd.DataFrame(per_sample)
    ps_path = results_path.parent / "calibration_per_sample.csv"
    ps_df.to_csv(ps_path, index=False)
    print(f"\\nPer-sample results persisted to {ps_path}")

    if len(y_true) > 1:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        err = y_pred - y_true
        mae = np.mean(np.abs(err))
        medae = np.median(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        bias = np.mean(err)
        
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (np.sum(err**2) / ss_tot) if ss_tot > 0 else float('nan')
        
        pearson, _ = pearsonr(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)
        
        print("\\n--- METRICS ---")
        print(f"MAE: {mae:.4f} eV")
        print(f"MedAE: {medae:.4f} eV")
        print(f"RMSE: {rmse:.4f} eV")
        print(f"Mean Bias: {bias:.4f} eV")
        print(f"R²: {r2:.4f}")
        print(f"Pearson: {pearson:.4f}")
        print(f"Spearman: {spearman:.4f}")
    else:
        print("\\nInsufficient data points for full metric calculation.")

if __name__ == "__main__":
    main()
