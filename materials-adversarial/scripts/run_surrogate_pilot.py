import os
import json
import pandas as pd
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.materials_adv.domain.chemistry.surrogate_geometry import SurrogateGeometryBuilder
from rdkit import Chem

def run_pilot():
    out_dir = Path("results/surrogate_structure_pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = "hpc_oracle/configs/surrogate_structure_protocol.yaml"
    builder = SurrogateGeometryBuilder(config_path)
    
    pilot_path = "hpc_oracle/manifests/calibration_pilot.csv"
    if not os.path.exists(pilot_path):
        print(f"Cannot find pilot dataset at {pilot_path}")
        return
        
    pilot_df = pd.read_csv(pilot_path)
    lengths = builder.config.get("oligomer_lengths", [2, 3])
    
    structure_manifest = []
    conformer_summaries = []
    job_manifest = []
    
    # Counter for total successes
    success_counts = {l: 0 for l in lengths}
    
    for row in pilot_df.itertuples():
        smiles = row.polymer_representation
        pilot_id = row.pilot_id
        
        row_status = {
            "pilot_id": pilot_id,
            "smiles": smiles,
            "periodic_surrogate_readiness": "BLOCKED_MISSING_CELL_PARAMS" 
        }
        
        for n in lengths:
            mol, uncertainty = builder.build_surrogate(smiles, oligomer_length=n)
            key = f"capped_n{n}"
            
            if mol is not None:
                row_status[f"{key}_success"] = True
                success_counts[n] += 1
                
                # Write SDF
                sdf_path = out_dir / f"{pilot_id}_{key}.sdf"
                writer = Chem.SDWriter(str(sdf_path))
                writer.write(mol)
                writer.close()
                
                # Job Manifest
                job_manifest.append({
                    "pilot_id": pilot_id,
                    "variant": key,
                    "structure_file": str(sdf_path),
                    "surrogate_type": "CALIBRATABLE_SURROGATE",
                    "uncertainty_hash": uncertainty["topology_hash"],
                    "oligomer_length": n,
                    "energy_conformer": uncertainty["selected_conformer_energy"]
                })
                
                conformer_summaries.append({
                    "pilot_id": pilot_id,
                    "variant": key,
                    "num_generated": uncertainty["num_conformers_generated"],
                    "energy_spread_std": uncertainty["energy_std"],
                    "selected_energy": uncertainty["selected_conformer_energy"]
                })
                
            else:
                row_status[f"{key}_success"] = False
                row_status[f"{key}_failure_reason"] = uncertainty.get("error", "Unknown")
                
        structure_manifest.append(row_status)
        
    pd.DataFrame(structure_manifest).to_csv(out_dir / "structure_manifest.csv", index=False)
    pd.DataFrame(conformer_summaries).to_csv(out_dir / "conformer_summary.csv", index=False)
    
    # Save the calibration jobs manifest
    manifest_dir = Path("hpc_oracle/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(job_manifest).to_csv(manifest_dir / "surrogate_calibration_jobs.csv", index=False)
    
    print("\n--- Surrogate Pilot Summary ---")
    print(f"Total Pilot Polymers: {len(pilot_df)}")
    for l, c in success_counts.items():
        print(f"n={l} Successes: {c}")
    print(f"Outputs written to {out_dir}")
    print(f"Job manifest written to {manifest_dir / 'surrogate_calibration_jobs.csv'}")

if __name__ == "__main__":
    run_pilot()
