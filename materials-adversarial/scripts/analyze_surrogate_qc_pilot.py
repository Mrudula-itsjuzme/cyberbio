import os
import pandas as pd
from pathlib import Path
import json

def analyze_pilot():
    manifest_path = "hpc_oracle/manifests/surrogate_qc_pilot_9jobs.csv"
    if not os.path.exists(manifest_path):
        print(f"Missing {manifest_path}")
        return
        
    jobs_df = pd.read_csv(manifest_path)
    
    out_dir = Path("results/surrogate_qc_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Since we are dry-running, the execution results are mock/blocked.
    results = []
    failures = []
    
    for _, row in jobs_df.iterrows():
        # Backend execution is blocked
        res = {
            "job_id": row["job_id"],
            "polymer_id": row["polymer_id"],
            "oligomer_length": row["oligomer_length"],
            "dataset_bandgap_eV": row["dataset_bandgap_eV"],
            "surrogate_gap_eV": None,
            "status": "BACKEND_EXECUTION_BLOCKED",
            "failure_reason": "No QE/VASP backend available"
        }
        results.append(res)
        failures.append(res)
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "raw_results.csv", index=False)
    
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(out_dir / "failures.csv", index=False)
    
    # Oligomer-length analysis (requires data, but we can structure the output)
    length_analysis = []
    for poly_id in jobs_df['polymer_id'].unique():
        poly_res = results_df[results_df['polymer_id'] == poly_id]
        if len(poly_res) == 0: continue
        
        gaps = {}
        for n in [2, 3, 4]:
            n_res = poly_res[poly_res['oligomer_length'] == n]
            gaps[n] = n_res['surrogate_gap_eV'].iloc[0] if len(n_res) > 0 else None
            
        diff_3_2 = abs(gaps[3] - gaps[2]) if gaps[3] is not None and gaps[2] is not None else None
        diff_4_3 = abs(gaps[4] - gaps[3]) if gaps[4] is not None and gaps[3] is not None else None
        
        length_analysis.append({
            "polymer_id": poly_id,
            "gap_n2": gaps[2],
            "gap_n3": gaps[3],
            "gap_n4": gaps[4],
            "abs_diff_n3_n2": diff_3_2,
            "abs_diff_n4_n3": diff_4_3,
            "trend": "INCONCLUSIVE"
        })
        
    pd.DataFrame(length_analysis).to_csv(out_dir / "oligomer_length_analysis.csv", index=False)
    
    # Dataset Alignment JSON
    dataset_alignment = {
        "status": "EXPLORATORY_ONLY",
        "n2": {"MAE": None, "RMSE": None, "R2": None, "Pearson": None, "Spearman": None},
        "n3": {"MAE": None, "RMSE": None, "R2": None, "Pearson": None, "Spearman": None},
        "n4": {"MAE": None, "RMSE": None, "R2": None, "Pearson": None, "Spearman": None}
    }
    
    with open(out_dir / "dataset_alignment.json", "w") as f:
        json.dump(dataset_alignment, f, indent=4)
        
    calibration_summary = {
        "total_jobs": len(jobs_df),
        "successful_jobs": 0,
        "failed_jobs": len(jobs_df),
        "verdict": "BACKEND_EXECUTION_BLOCKED"
    }
    
    with open(out_dir / "calibration_summary.json", "w") as f:
        json.dump(calibration_summary, f, indent=4)
        
    print("Generated calibration reports with BACKEND_EXECUTION_BLOCKED status.")

if __name__ == "__main__":
    analyze_pilot()
