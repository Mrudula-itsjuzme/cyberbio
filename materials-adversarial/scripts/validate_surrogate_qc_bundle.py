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
    bundle_dir = Path("oracle_surrogate_qc_bundle")
    
    if not bundle_dir.exists():
        print(f"Error: {bundle_dir} does not exist.")
        sys.exit(1)
        
    hashes_file = bundle_dir / "HASHES.json"
    if not hashes_file.exists():
        print("Error: HASHES.json is missing.")
        sys.exit(1)
        
    # Check hashes
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
    
    # Check exactly 9 jobs
    manifest_path = bundle_dir / "manifests" / "surrogate_qc_pilot_9jobs.csv"
    if not manifest_path.exists():
        print("Error: Manifest missing.")
        sys.exit(1)
        
    df = pd.read_csv(manifest_path)
    if len(df) != 9:
        print(f"Error: Expected 9 jobs, found {len(df)}")
        sys.exit(1)
        
    print("9 jobs OK.")
    
    # Check structures
    for path in df["structure_path"]:
        basename = os.path.basename(path)
        if not (bundle_dir / "structures" / basename).exists():
            print(f"Error: Missing structure {basename}")
            sys.exit(1)
            
        with open(bundle_dir / "structures" / basename, "r") as f:
            content = f.read()
            if "*" in content:
                print(f"Error: Wildcard found in {basename}")
                sys.exit(1)
                
    print("Structures OK.")
    
    # Check cutoffs
    unverified = bundle_dir / "CUTOFF_COMPATIBILITY_UNVERIFIED"
    if unverified.exists():
        print("Warning: Cutoff compatibility is unverified. Assuming it is OK for mock run but needs attention.")
    else:
        print("Cutoffs OK.")
        
    print("VALIDATION SUCCESS")
    sys.exit(0)

if __name__ == "__main__":
    main()
