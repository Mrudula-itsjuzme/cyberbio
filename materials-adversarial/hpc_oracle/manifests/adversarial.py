import hashlib
import json
from typing import Dict, Any, List

def generate_adversarial_manifest(shortlist: List[Dict[str, Any]], config_hash: str, backend: str) -> List[Dict[str, Any]]:
    jobs = []
    
    for pair in shortlist:
        source_smi = pair.get("source_smiles")
        cand_smi = pair.get("candidate_smiles")
        
        # We need independent jobs for source and candidate
        source_job_id = hashlib.md5(f"adv_{source_smi}_{backend}_{config_hash}".encode()).hexdigest()
        cand_job_id = hashlib.md5(f"adv_{cand_smi}_{backend}_{config_hash}".encode()).hexdigest()
        
        pair_id = hashlib.md5(f"{source_job_id}_{cand_job_id}".encode()).hexdigest()
        
        jobs.append({
            "pair_id": pair_id,
            "source_job": source_job_id,
            "candidate_job": cand_job_id,
            "edit_count": pair.get("edit_count"),
            "attack_method": pair.get("attack_method"),
            "query_budget": pair.get("query_budget"),
            "GraphMPNN_delta": pair.get("GraphMPNN_delta"),
            "Transformer_delta": pair.get("Transformer_delta"),
            "trajectory_hash": pair.get("trajectory_hash"),
            "backend": backend,
            "config_hash": config_hash
        })
        
    return jobs
