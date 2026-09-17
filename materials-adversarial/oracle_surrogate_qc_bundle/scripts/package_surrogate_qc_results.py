import os
import json
import hashlib
import shutil
from pathlib import Path
import pandas as pd

def get_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def package_results():
    bundle_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(bundle_dir)
    
    out_bundle = Path("surrogate_qc_results_bundle")
    out_bundle.mkdir(exist_ok=True)
    
    # Copy essential metadata
    for file in ["cluster_environment.json", "cluster_pseudopotential_manifest.json", "configs/SURROGATE_PROTOCOL_LOCK.json", "manifests/surrogate_qc_pilot_9jobs.csv", "HASHES.json"]:
        p = Path(file)
        if p.exists():
            dest = out_bundle / p.name
            shutil.copy(p, dest)
            
    # Compile job statuses
    manifest_csv = "manifests/surrogate_qc_pilot_9jobs.csv"
    if Path(manifest_csv).exists():
        df = pd.read_csv(manifest_csv)
        statuses = []
        for job_id in df["job_id"]:
            out_dir = Path("out") / job_id
            status_file = out_dir / "status.txt"
            status = "UNKNOWN_FAILURE"
            if status_file.exists():
                status = status_file.read_text().strip()
                
            exit_file = out_dir / "exit_status.json"
            exit_code = -1
            if exit_file.exists():
                try:
                    exit_code = json.loads(exit_file.read_text()).get("exit_code", -1)
                except:
                    pass
                    
            statuses.append({
                "job_id": job_id,
                "status": status,
                "exit_code": exit_code
            })
            
        pd.DataFrame(statuses).to_csv(out_bundle / "job_statuses.csv", index=False)
        
    # Copy out directory
    if Path("out").exists():
        dest_out = out_bundle / "out"
        if dest_out.exists():
            shutil.rmtree(dest_out)
        shutil.copytree("out", dest_out)
        
    # Generate hashes for result bundle
    hashes = {}
    for root, dirs, files in os.walk(out_bundle):
        for file in files:
            if file == "RESULT_HASHES.json": continue
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, out_bundle)
            hashes[rel_path] = get_hash(filepath)
            
    with open(out_bundle / "RESULT_HASHES.json", "w") as f:
        json.dump(hashes, f, indent=4)
        
    print(f"Packaged results into {out_bundle}")

if __name__ == "__main__":
    package_results()
