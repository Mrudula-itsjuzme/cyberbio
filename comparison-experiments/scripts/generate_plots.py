import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

def main():
    raw_dir = Path("/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/raw")
    plots_dir = Path("/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    for fpath in raw_dir.glob("raw_*.jsonl"):
        with open(fpath, "r") as f:
            for line in f:
                data = json.loads(line)
                records.append(data)
                
    if not records:
        print("No raw results found to plot.")
        return
        
    df = pd.DataFrame(records)
    
    # Plot 1: Drift distribution by attacker
    plt.figure(figsize=(10, 6))
    df.boxplot(column="prediction_drift", by="attack_name", grid=False)
    plt.title("Prediction Drift Distribution by Attacker")
    plt.suptitle("")
    plt.savefig(plots_dir / "drift_distribution.png")
    plt.close('all')
    
    # Plot 2: Validity rate
    df_valid = df.groupby("attack_name")["constraint_pass"].mean()
    plt.figure(figsize=(8, 5))
    df_valid.plot(kind="bar")
    plt.title("Constraint Pass Rate by Attacker")
    plt.ylabel("Rate")
    plt.savefig(plots_dir / "validity_rate.png")
    plt.close('all')

    # Plot 3: Drift vs Similarity
    plt.figure(figsize=(8, 6))
    for name, group in df.groupby("attack_name"):
        plt.scatter(group["tanimoto_similarity"], group["prediction_drift"], label=name, alpha=0.7)
    plt.title("Drift vs Tanimoto Similarity")
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Prediction Drift")
    plt.legend()
    plt.savefig(plots_dir / "drift_vs_similarity.png")
    plt.close('all')

    # Plot 4: Runtime vs Drift
    plt.figure(figsize=(8, 6))
    for name, group in df.groupby("attack_name"):
        plt.scatter(group["runtime_seconds"], group["prediction_drift"], label=name, alpha=0.7)
    plt.title("Runtime vs Drift")
    plt.xlabel("Runtime (s)")
    plt.ylabel("Prediction Drift")
    plt.legend()
    plt.savefig(plots_dir / "runtime_vs_drift.png")
    plt.close('all')

    print(f"Generated {len(df)} plots in {plots_dir}")

if __name__ == "__main__":
    main()
