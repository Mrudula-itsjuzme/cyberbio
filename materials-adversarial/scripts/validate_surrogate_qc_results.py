import os
import sys
import json
import hashlib
import pandas as pd
from pathlib import Path

def get_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    bundle_dir = Path("oracle_surrogate_qc_bundle/surrogate_qc_results_bundle")
    
    if not bundle_dir.exists():
        print(f"Error: {bundle_dir} does not exist.")
        sys.exit(1)
        
    hashes_file = bundle_dir / "RESULT_HASHES.json"
    if not hashes_file.exists():
        print("Error: RESULT_HASHES.json is missing.")
        sys.exit(1)
        
    with open(hashes_file) as f:
        hashes = json.load(f)
        
    for rel_path, expected_hash in hashes.items():
        filepath = bundle_dir / rel_path
        if not filepath.exists():
            print(f"Error: Missing file {rel_path}")
            sys.exit(1)
        
        actual_hash = get_hash(filepath)
        if actual_hash != expected_hash:
            print(f"Error: Hash mismatch for {rel_path}")
            sys.exit(1)
            
    print("Hashes OK.")
    
    # Check protocol mismatch
    local_lock = Path("hpc_oracle/configs/SURROGATE_PROTOCOL_LOCK.json")
    remote_lock = bundle_dir / "SURROGATE_PROTOCOL_LOCK.json"
    
    if not remote_lock.exists():
        print("Error: Remote lock missing.")
        sys.exit(1)
        
    if get_hash(local_lock) != get_hash(remote_lock):
        print("Error: Protocol mismatch! The physics configuration returned from the cluster does not match the local expectation.")
        sys.exit(1)
        
    print("Protocol matches expected version.")
    
    # Check jobs
    manifest = bundle_dir / "surrogate_qc_pilot_9jobs.csv"
    if not manifest.exists():
        print("Error: Manifest missing.")
        sys.exit(1)
        
    df = pd.read_csv(manifest)
    for job_id in df["job_id"]:
        out_dir = bundle_dir / "out" / job_id
        if not out_dir.exists():
            print(f"Error: Missing output directory for {job_id}")
            sys.exit(1)
            
    # Check environment
    if not (bundle_dir / "cluster_environment.json").exists():
        print("Error: Missing cluster environment snapshot.")
        sys.exit(1)
        
    if not (bundle_dir / "cluster_pseudopotential_manifest.json").exists():
        print("Error: Missing pseudopotential manifest.")
        sys.exit(1)
        
    print("VALIDATION SUCCESS")
    sys.exit(0)

if __name__ == "__main__":
    main()
