import os
import pandas as pd
from pathlib import Path
from materials_adv.oracle.surrogate_parser import SurrogateParser
import json

def run_synthetic_pipeline():
    out_dir = Path("oracle_surrogate_qc_bundle/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate some synthetic job definitions
    jobs = []
    poly_ids = ["poly_A", "poly_B", "poly_C"]
    
    parser = SurrogateParser()
    
    results = []
    
    # We create 9 mock outputs
    mock_dir = Path("oracle_surrogate_qc_bundle/mock_outputs")
    mock_dir.mkdir(exist_ok=True)
    
    for i, poly in enumerate(poly_ids):
        for n in [2, 3, 4]:
            job_id = f"{poly}_n{n}"
            out_file = mock_dir / f"{job_id}.out"
            
            # Write a synthetic QE output
            gap = 1.0 + i + (1.0 / n) # Some synthetic trend
            
            with open(out_file, "w") as f:
                f.write(f"""
     Program PWSCF v.6.8 starts on 14Sep2026 at 10:00:00 
     !    total energy              =     -123.45678900 Ry
     highest occupied, lowest unoccupied level (ev):     -5.0000    {gap - 5.0:.4f}
     JOB DONE.
                """)
                
            res = parser.parse_qe_output(str(out_file))
            res["job_id"] = job_id
            res["polymer_id"] = poly
            res["oligomer_length"] = n
            res["input_hash"] = "mock_hash"
            res["output_hash"] = "mock_hash"
            res["protocol_hash"] = "mock_hash"
            results.append(res)
            
    df = pd.DataFrame(results)
    
    # Reorder columns per Result Ingestion Contract
    cols = ["job_id", "polymer_id", "oligomer_length", "status", "converged", 
            "surrogate_gap_eV", "HOMO_or_VBM_eV", "LUMO_or_CBM_eV", "total_energy_eV",
            "SCF_iterations", "runtime", "warnings", "failure_reason", "input_hash",
            "output_hash", "protocol_hash"]
    df = df[cols]
    
    df.to_csv(out_dir / "synthetic_results.csv", index=False)
    
    # Analysis
    length_analysis = []
    for poly_id in poly_ids:
        poly_res = df[df['polymer_id'] == poly_id]
        
        gaps = {}
        for n in [2, 3, 4]:
            n_res = poly_res[poly_res['oligomer_length'] == n]
            gaps[n] = n_res['surrogate_gap_eV'].iloc[0] if len(n_res) > 0 else None
            
        diff_3_2 = abs(gaps[3] - gaps[2]) if gaps[3] is not None and gaps[2] is not None else None
        diff_4_3 = abs(gaps[4] - gaps[3]) if gaps[4] is not None and gaps[3] is not None else None
        
        length_analysis.append({
            "polymer_id": poly_id,
            "gap_n2": gaps[2],
            "gap_n3": gaps[3],
            "gap_n4": gaps[4],
            "abs_diff_n3_n2": diff_3_2,
            "abs_diff_n4_n3": diff_4_3,
            "trend": "SYNTHETIC_TEST_TREND"
        })
        
    pd.DataFrame(length_analysis).to_csv(out_dir / "synthetic_oligomer_analysis.csv", index=False)
    
    print("SYNTHETIC PIPELINE TEST SUCCESS")
    
if __name__ == "__main__":
    run_synthetic_pipeline()
