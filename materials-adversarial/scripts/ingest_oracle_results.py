import argparse
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from hpc_oracle.backend import QuantumEspressoBackend, VASPBackend

def main():
    parser = argparse.ArgumentParser(description="Ingest Oracle Results")
    parser.add_argument("--export-dir", type=str, required=True, help="Directory containing HPC export")
    parser.add_argument("--backend", type=str, required=True, choices=["qe", "vasp"])
    args = parser.parse_args()
    
    export_dir = Path(args.export_dir)
    manifest_path = export_dir / "manifest.json"
    
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        sys.exit(1)
        
    with open(manifest_path) as f:
        jobs = json.load(f)
        
    if args.backend == "qe":
        backend = QuantumEspressoBackend()
    else:
        backend = VASPBackend()
        
    ingested = []
    
    for job in jobs:
        job_dir = export_dir / "jobs" / job["job_id"]
        parsed = backend.parse_output(str(job_dir))
        
        job["status"] = "COMPLETED" if parsed.converged else "FAILED"
        job["converged"] = parsed.converged
        job["bandgap_eV"] = parsed.bandgap_eV
        job["warnings"] = parsed.warnings
        job["failure_reason"] = parsed.failure_reason
        
        ingested.append(job)
        
    out_file = export_dir / "ingested_results.json"
    with open(out_file, "w") as f:
        json.dump(ingested, f, indent=2)
        
    print(f"Ingested {len(ingested)} jobs to {out_file}")

if __name__ == "__main__":
    main()
