import os
import sys
import pandas as pd
from pathlib import Path
from materials_adv.oracle.surrogate_parser import SurrogateParser

def main():
    bundle_dir = Path("oracle_surrogate_qc_bundle/surrogate_qc_results_bundle")
    
    if not bundle_dir.exists():
        print(f"Error: {bundle_dir} does not exist.")
        sys.exit(1)
        
    manifest = bundle_dir / "surrogate_qc_pilot_9jobs.csv"
    if not manifest.exists():
        print("Error: Manifest missing.")
        sys.exit(1)
        
    df = pd.read_csv(manifest)
    
    results = []
    
    parser = SurrogateParser()
    for _, row in df.iterrows():
        job_id = row["job_id"]
        out_file = bundle_dir / "out" / job_id / "qe.out"
        
        parsed = parser.parse_qe_output(str(out_file))
        
        res = {
            "job_id": job_id,
            "polymer_id": row["polymer_id"],
            "oligomer_length": row["oligomer_length"],
            "gap_ev": parsed.get("band_gap_ev", None),
            "total_energy_ryd": parsed.get("total_energy_ryd", None),
            "is_converged": parsed.get("is_converged", False)
        }
        results.append(res)
        
    res_df = pd.DataFrame(results)
    
    out_dir = Path("results/surrogate_qc_ingested")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_df.to_csv(out_dir / "parsed_gaps.csv", index=False)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
