import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    base_dir = Path(__file__).resolve().parent.parent / "results/raw/llm"
    out_dir = Path(__file__).resolve().parent.parent / "results/summaries/llm"
    plot_dir = Path(__file__).resolve().parent.parent / "results/plots/llm"
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist. Run LLM benchmark first.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    for fpath in base_dir.rglob("raw_*.jsonl"):
        parts = fpath.parts
        budget = int([p for p in parts if p.startswith("budget_")][0].split("_")[1])
        seed = int([p for p in parts if p.startswith("seed_")][0].split("_")[1])
        
        with open(fpath, "r") as fin:
            for line in fin:
                data = json.loads(line)
                data["query_budget"] = budget
                data["seed"] = seed
                all_results.append(data)
                
    if not all_results:
        print("No results found.")
        return
        
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "llm_sweep_results.csv", index=False)
    
    # Calculate unique proposal ratio and efficiencies
    df["is_success"] = df["prediction_drift"] > 0.0
    
    # Generate failure mode table
    failures = df["failure_reason"].value_counts().reset_index()
    failures.columns = ["failure_reason", "count"]
    failures.to_csv(out_dir / "failure_modes.csv", index=False)
    
    print("Aggregated LLM results and failure modes.")
    
if __name__ == "__main__":
    main()
