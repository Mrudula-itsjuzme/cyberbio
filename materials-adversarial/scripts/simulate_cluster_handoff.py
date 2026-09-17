import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    print("=== STARTING SYNTHETIC CLUSTER HANDOFF TEST ===")
    
    repo_root = Path(__file__).parent.parent
    bundle_dir = repo_root / "oracle_surrogate_qc_bundle"
    
    # 1. Run Cluster Verification
    print("\n--- Running Cluster Verification ---")
    res = subprocess.run([sys.executable, str(bundle_dir / "scripts" / "verify_cluster_environment.py")])
    if res.returncode != 0:
        print("Cluster verification failed (expected if pseudopotentials are truly missing).")
        print("Mocking success for synthetic pipeline...")
        # Mocking
        manifest = {
            "C": {"status": "FOUND", "filename": "C_sssp_eff_1.2.upf", "hash": "mock", "suggested_cutoff_Ry": 30.0},
            "H": {"status": "FOUND", "filename": "H_sssp_eff_1.2.upf", "hash": "mock", "suggested_cutoff_Ry": 35.0},
            "N": {"status": "FOUND", "filename": "N_sssp_eff_1.2.upf", "hash": "mock", "suggested_cutoff_Ry": 30.0},
            "O": {"status": "FOUND", "filename": "O_sssp_eff_1.2.upf", "hash": "mock", "suggested_cutoff_Ry": 35.0},
            "S": {"status": "FOUND", "filename": "S_sssp_eff_1.2.upf", "hash": "mock", "suggested_cutoff_Ry": 30.0}
        }
        import json
        with open(bundle_dir / "cluster_pseudopotential_manifest.json", "w") as f:
            json.dump(manifest, f)
            
        env = {
            "hostname": "synthetic-cluster",
            "qe_executable_path": "/mock/pw.x",
            "qe_version": "v7.1",
            "scheduler_type": "SLURM"
        }
        with open(bundle_dir / "cluster_environment.json", "w") as f:
            json.dump(env, f)
            
    # 2. Mock execution
    print("\n--- Mocking Cluster Execution ---")
    import pandas as pd
    df = pd.read_csv(bundle_dir / "manifests" / "surrogate_qc_pilot_9jobs.csv")
    out_dir = bundle_dir / "out"
    out_dir.mkdir(exist_ok=True)
    
    mock_out_content = """
     Program PWSCF v.7.0 (svn rev. 0) starts on  1Jan1970 at  0:00: 0
     !    total energy              =     -10.00000000 ryd
     highest occupied, lowest unoccupied level (ev):     1.0000    2.0000
     JOB DONE.
"""
    
    for job_id in df["job_id"]:
        job_out = out_dir / job_id
        job_out.mkdir(exist_ok=True)
        with open(job_out / "qe.out", "w") as f:
            f.write(mock_out_content)
        with open(job_out / "status.txt", "w") as f:
            f.write("FINISHED")
        with open(job_out / "exit_status.json", "w") as f:
            f.write('{"exit_code": 0}')
            
    # 3. Package Results
    print("\n--- Packaging Results ---")
    subprocess.run([sys.executable, str(bundle_dir / "scripts" / "package_surrogate_qc_results.py")], check=True)
    
    # 4. Validate Returned Results
    print("\n--- Validating Returned Results ---")
    subprocess.run([sys.executable, str(repo_root / "scripts" / "validate_surrogate_qc_results.py")], check=True)
    
    # 5. Ingest Results
    print("\n--- Ingesting Results ---")
    os.environ["PYTHONPATH"] = "src"
    subprocess.run([sys.executable, str(repo_root / "scripts" / "ingest_surrogate_qc_results.py")], check=True, env=os.environ)
    
    print("\n=== SYNTHETIC CLUSTER HANDOFF TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
