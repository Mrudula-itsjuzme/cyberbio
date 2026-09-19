import argparse
import itertools
import subprocess
import os
import json
import pandas as pd
from pathlib import Path

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. The real provider will fail. Exiting.")
        # We can either exit or continue with the knowledge it will fail.
        # Given the instruction "run a tiny real-provider pilot", I'll run it, let it fail natively, and log the failure.
    
    script = Path(__file__).resolve().parent / "run_comparison.py"
    base_out = Path(__file__).resolve().parent.parent / "results/raw/llm_pilot"
    
    # Tiny real provider pilot
    modes = ["objective-aware", "iterative"]
    budget = 10
    seed = 42
    source_count = 5
    
    for mode in modes:
        out_dir = base_out / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python", str(script),
            "--attack", f"llm_{mode}",
            "--source-count", str(source_count),
            "--query-budget", str(budget),
            "--seed", str(seed),
            "--output-dir", str(out_dir), "--use-real-llm"
        ]
        
        print(f"Running LLM Pilot: {mode}")
        subprocess.run(cmd)

    # Compile the manual inspection table
    table = []
    for fpath in base_out.rglob("raw_*.jsonl"):
        with open(fpath, "r") as fin:
            for line in fin:
                data = json.loads(line)
                
                # Fetch history / rationale
                hist = data.get("metadata", {}).get("history", [])
                edit_desc = "none"
                # If there's an actual candidate selected:
                if hist and data["candidate_sequence"] != data["source_sequence"]:
                    # Try to extract the best description if we saved it... wait we only saved rationale
                    pass 
                
                table.append({
                    "mode": data["prompt_mode"],
                    "source_id": data["source_id"],
                    "source_sequence": data["source_sequence"],
                    "LLM proposal": data["candidate_sequence"],
                    "rationale": data.get("metadata", {}).get("rationale", ""),
                    "RDKit validity": data.get("valid_rdkit", False),
                    "constraint result": data.get("constraint_pass", False),
                    "Tanimoto": data.get("tanimoto_similarity", 0.0),
                    "edit distance": data.get("edit_distance", 0),
                    "source prediction": data.get("source_prediction", 0.0),
                    "candidate prediction": data.get("candidate_prediction", 0.0),
                    "drift": data.get("prediction_drift", 0.0),
                    "failure_reason": data.get("failure_reason", "")
                })
                
    if table:
        df = pd.DataFrame(table)
        print("\n--- LLM PILOT INSPECTION TABLE ---")
        print(df.to_string())
        df.to_csv(base_out.parent.parent / "summaries" / "llm_pilot_inspection.csv", index=False)

if __name__ == "__main__":
    main()
