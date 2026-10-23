import json
import pandas as pd
from pathlib import Path
import numpy as np

def main():
    base_out = Path(__file__).resolve().parent.parent / "results/raw/budget_sweep"
    summaries_dir = Path(__file__).resolve().parent.parent / "results/summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for fpath in base_out.rglob("raw_*.jsonl"):
        parts = fpath.parts
        budget_str = [p for p in parts if p.startswith("budget_") and p != "budget_sweep"][0]
        seed_str = [p for p in parts if p.startswith("seed_")][0]
        budget = int(budget_str.split("_")[1])
        seed = int(seed_str.split("_")[1])
        
        with open(fpath, "r") as f:
            for line in f:
                data = json.loads(line)
                data["query_budget"] = budget
                data["seed"] = seed
                all_results.append(data)
                
    if not all_results:
        print("No raw results found.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Save full canonical machine-readable file
    df.to_csv(summaries_dir / "budget_sweep_results.csv", index=False)
    with open(summaries_dir / "budget_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Aggregated {len(df)} results to budget_sweep_results.csv")
    
if __name__ == "__main__":
    main()
