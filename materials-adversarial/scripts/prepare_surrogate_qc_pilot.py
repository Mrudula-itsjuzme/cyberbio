import os
import pandas as pd
from pathlib import Path
import yaml
import shutil

def prepare_pilot():
    manifest_path = "hpc_oracle/manifests/surrogate_calibration_jobs.csv"
    if not os.path.exists(manifest_path):
        print(f"Missing {manifest_path}")
        return
        
    jobs_df = pd.read_csv(manifest_path)
    
    # We need dataset bandgap to pick low, medium, high.
    # We can get that from the pilot calibration file.
    pilot_csv = "hpc_oracle/manifests/calibration_pilot.csv"
    if not os.path.exists(pilot_csv):
        print("Missing pilot targets.")
        return
        
    targets_df = pd.read_csv(pilot_csv)
    
    # Merge targets
    merged = jobs_df.merge(targets_df[['pilot_id', 'dataset_bandgap_eV']], on='pilot_id')
    
    # Select 3 polymers
    unique_polymers = merged[['pilot_id', 'dataset_bandgap_eV']].drop_duplicates().sort_values('dataset_bandgap_eV')
    
    if len(unique_polymers) < 3:
        print("Not enough unique polymers.")
        return
        
    # Low, medium, high
    low_poly = unique_polymers.iloc[0]['pilot_id']
    med_poly = unique_polymers.iloc[len(unique_polymers)//2]['pilot_id']
    high_poly = unique_polymers.iloc[-1]['pilot_id']
    
    selected = [low_poly, med_poly, high_poly]
    categories = {low_poly: "low_gap", med_poly: "medium_gap", high_poly: "high_gap"}
    
    selected_jobs = merged[merged['pilot_id'].isin(selected)].copy()
    
    out_jobs = []
    job_counter = 0
    
    for _, row in selected_jobs.iterrows():
        # Validate structure (mock check - assumes previous stage succeeded and .sdf exists)
        if not os.path.exists(row['structure_file']):
            status = "INPUT_INVALID"
            reason = "Structure file missing"
        else:
            status = "DRY_RUN_PREPARED"
            reason = ""
            
        out_jobs.append({
            "job_id": f"surrogate_qc_{job_counter:03d}",
            "polymer_id": row['pilot_id'],
            "gap_category": categories[row['pilot_id']],
            "oligomer_length": row['oligomer_length'],
            "dataset_bandgap_eV": row['dataset_bandgap_eV'],
            "structure_path": row['structure_file'],
            "structure_hash": "not_hashed_yet",
            "topology_hash": row['uncertainty_hash'],
            "selected_conformer_energy": row['energy_conformer'],
            "backend": "Quantum ESPRESSO",
            "config_hash": "config_hash_placeholder",
            "status": status,
            "failure_reason": reason
        })
        job_counter += 1
        
    out_df = pd.DataFrame(out_jobs)
    out_df.to_csv("hpc_oracle/manifests/surrogate_qc_pilot_9jobs.csv", index=False)
    
    print(f"Prepared {len(out_df)} jobs in surrogate_qc_pilot_9jobs.csv")

if __name__ == "__main__":
    prepare_pilot()
