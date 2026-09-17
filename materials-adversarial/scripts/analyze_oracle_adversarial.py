import argparse
import sys
import json
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze Oracle Adversarial Errors")
    parser.add_argument("--ingested-results", type=str, required=True, help="Path to ingested_results.json (adversarial)")
    args = parser.parse_args()
    
    results_path = Path(args.ingested_results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)
        
    with open(results_path) as f:
        jobs = json.load(f)
        
    # We need to map jobs by their job_id
    job_map = {j["job_id"]: j for j in jobs if "job_id" in j}
    
    print("=== ADVERSARIAL ORACLE ANALYSIS ===")
    
    # We also need the pair manifest to know which job is source and which is candidate.
    # Assuming the jobs array itself contains pair_id, source_job, candidate_job for the pair entries
    # Wait, in export_oracle_jobs.py we didn't export the pair manifest specifically, but we can assume
    # the jobs themselves carry pair info if it's the adversarial manifest.
    # Let's just output the expected math for pairs that we can match.
    
    evaluated_pairs = 0
    for j in jobs:
        if "pair_id" in j:
            source_id = j.get("source_job")
            cand_id = j.get("candidate_job")
            
            if source_id in job_map and cand_id in job_map:
                s_job = job_map[source_id]
                c_job = job_map[cand_id]
                
                if s_job.get("converged") and c_job.get("converged"):
                    t_source = s_job.get("bandgap_eV", 0)
                    t_cand = c_job.get("bandgap_eV", 0)
                    
                    delta_t = t_cand - t_source
                    delta_g = j.get("GraphMPNN_delta", 0)
                    delta_s = j.get("Transformer_delta", 0)
                    
                    e_g = abs(delta_g - delta_t)
                    e_s = abs(delta_s - delta_t)
                    
                    print(f"Pair: {j['pair_id']}")
                    print(f"  Delta_T: {delta_t:.4f}")
                    print(f"  Delta_G: {delta_g:.4f} -> E_G: {e_g:.4f}")
                    print(f"  Delta_S: {delta_s:.4f} -> E_S: {e_s:.4f}")
                    evaluated_pairs += 1
                    
    if evaluated_pairs == 0:
        print("No completed pairs found for analysis.")
    else:
        print(f"\\nEvaluated {evaluated_pairs} adversarial pairs.")
    
if __name__ == "__main__":
    main()
