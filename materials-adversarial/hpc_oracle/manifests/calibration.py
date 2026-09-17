import hashlib
import json
import pandas as pd
from typing import Dict, Any, List

def generate_calibration_manifest(df: pd.DataFrame, config_hash: str, backend: str) -> List[Dict[str, Any]]:
    jobs = []
    for _, row in df.iterrows():
        smiles = row.get("original_representation", "")
        # Deterministic job id
        job_id = hashlib.md5(f"calib_{smiles}_{backend}_{config_hash}".encode()).hexdigest()
        
        jobs.append({
            "job_id": job_id,
            "source_id": smiles,
            "dataset_bandgap_eV": row.get("property_value"),
            "structure_path": f"structures/calib_{job_id}.xyz", # Skeleton path
            "backend": backend,
            "config_hash": config_hash,
            "status": "PENDING",
            "failure_reason": None
        })
    return jobs
