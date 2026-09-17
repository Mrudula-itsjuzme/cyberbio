import argparse
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import shutil
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from hpc_oracle.backend import QuantumEspressoBackend, VASPBackend
from hpc_oracle.manifests.calibration import generate_calibration_manifest

def main():
    parser = argparse.ArgumentParser(description="Export Oracle Jobs (Dry Run)")
    parser.add_argument("--backend", type=str, required=True, choices=["qe", "vasp"])
    parser.add_argument("--subset", type=str, required=True, choices=["calibration", "adversarial", "calibration_pilot"])
    parser.add_argument("--dry-run", action="store_true", required=True, help="Must be true for now")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    
    if args.subset == "calibration_pilot":
        config_path = root / f"hpc_oracle/configs/calibration_pilot_{args.backend}.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {"pseudo_dir": "UNKNOWN", "functional": "UNKNOWN"}
    else:
        config = {"pseudo_dir": "dummy", "functional": "dummy", "cutoff_wfc": "dummy", "encut": "dummy"}
        
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
    
    if args.backend == "qe":
        backend = QuantumEspressoBackend(config)
    else:
        backend = VASPBackend(config)
        
    out_dir = root / f"oracle_{args.subset}_bundle" if args.subset == "calibration_pilot" else root / f"hpc_oracle_export_{args.backend}_{args.subset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Bundle README
    with open(out_dir / "README.md", "w") as f:
        f.write(f"# HPC Oracle Export: {args.subset.upper()}\\nBackend: {args.backend}\\n")
        if args.subset == "calibration_pilot":
            f.write("\\nNote: This bundle was generated as a pilot. Structure generation is BLOCKED as no explicit geometry was provided.\\n")
        
    if args.subset in ["calibration", "calibration_pilot"]:
        if args.subset == "calibration":
            df = pd.read_csv(root / "results/phase13_oracle_design/run_1/calibration_manifest.csv")
        else:
            df = pd.read_csv(root / "hpc_oracle/manifests/calibration_pilot.csv")
            # copy manifest
            shutil.copy(root / "hpc_oracle/manifests/calibration_pilot.csv", out_dir / "calibration_pilot.csv")
            # copy config
            config_out = out_dir / "config"
            config_out.mkdir(exist_ok=True)
            if config_path.exists():
                shutil.copy(config_path, config_out / config_path.name)
            
        jobs = generate_calibration_manifest(df, config_hash, args.backend)
        
        with open(out_dir / "hashes.json", "w") as f:
            json.dump({"config_hash": config_hash, "job_count": len(jobs)}, f, indent=2)
            
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(jobs, f, indent=2)
            
        # Write inputs
        for job in jobs:
            job_dir = out_dir / args.backend / job["job_id"]
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Write a marker file indicating structure is blocked for the pilot
            if args.subset == "calibration_pilot":
                with open(job_dir / "STRUCTURE_BLOCKED.txt", "w") as f:
                    f.write("Calculation blocked: explicit 3D geometry is required but not provided in dataset.\\n")
            else:
                structure_data = {"source_path": f"{job['source_id']}"}
                backend.write_input(structure_data, config, str(job_dir))
            
    print(f"Exported {args.subset} jobs for {args.backend} to {out_dir}")
    print("DRY-RUN: No execution performed.")

if __name__ == "__main__":
    main()
