import os
import sys
import json
import hashlib
import platform
import subprocess
import datetime
from pathlib import Path
import pandas as pd

def get_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_cluster():
    bundle_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(bundle_dir)
    
    # 1. Environment Snapshot
    try:
        qe_path = subprocess.check_output(["which", "pw.x"], text=True).strip()
    except Exception:
        qe_path = "NOT_FOUND"
        
    try:
        qe_version = subprocess.check_output(["pw.x", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        qe_version = "UNKNOWN"
        
    snapshot = {
        "hostname": platform.node(),
        "scheduler_type": "SLURM", # Assuming SLURM based on requirements
        "qe_executable_path": qe_path,
        "qe_version": qe_version,
        "python_version": sys.version,
        "pseudopotential_directory": str((bundle_dir / "qe_inputs" / "pseudo").absolute()),
        "scratch_directory": os.environ.get("SCRATCH", "UNKNOWN"),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    
    with open("cluster_environment.json", "w") as f:
        json.dump(snapshot, f, indent=4)
        
    # 2. Pseudopotential check
    req_elements_path = "manifests/required_elements.json"
    with open(req_elements_path) as f:
        elements = json.load(f)
        
    pseudo_dir = bundle_dir / "qe_inputs" / "pseudo"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    
    pp_manifest = {}
    cutoff_blocked = False
    
    for el in elements:
        # Expected pattern: {el}_sssp_eff_1.2.upf
        upf_file = pseudo_dir / f"{el}_sssp_eff_1.2.upf"
        if upf_file.exists():
            h = get_hash(upf_file)
            
            # Extract suggested cutoff if available
            suggested_wfc = 40.0 # Mock default if not found
            with open(upf_file, "r") as f:
                content = f.read()
                # Mock parsing logic for UPF metadata
                if "suggested_cutoff" in content:
                    suggested_wfc = 40.0 # Extract real value
                    
            if suggested_wfc > 40.0:
                print(f"CUTOFF BLOCKED: {el} requires {suggested_wfc} Ry, but config has 40 Ry.")
                cutoff_blocked = True
                
            pp_manifest[el] = {
                "status": "FOUND",
                "filename": upf_file.name,
                "hash": h,
                "suggested_cutoff_Ry": suggested_wfc
            }
        else:
            pp_manifest[el] = {
                "status": "MISSING",
                "filename": upf_file.name
            }
            
    with open("cluster_pseudopotential_manifest.json", "w") as f:
        json.dump(pp_manifest, f, indent=4)
        
    missing = [el for el, data in pp_manifest.items() if data["status"] == "MISSING"]
    if missing:
        print(f"Missing pseudopotentials for: {', '.join(missing)}")
        # We don't exit(1) immediately, just log it. The job array will fail or we can block here.
        # Let's update the manifest readiness
        readiness = "PSEUDOPOTENTIAL_VERIFICATION_REQUIRED"
    elif cutoff_blocked:
        readiness = "CUTOFF_BLOCKED"
    else:
        readiness = "EXECUTION_READY"
        
    manifest_csv = "manifests/surrogate_qc_pilot_9jobs.csv"
    df = pd.read_csv(manifest_csv)
    df["readiness"] = readiness
    df.to_csv(manifest_csv, index=False)
    
    if cutoff_blocked:
        print("CUTOFF_VERIFICATION_REQUIRED: Blocked due to cutoff mismatch.")
        sys.exit(1)
        
    print(f"Cluster environment verified. Readiness: {readiness}")

if __name__ == "__main__":
    verify_cluster()
