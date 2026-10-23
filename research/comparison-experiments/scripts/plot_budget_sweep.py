import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    summaries_dir = Path(__file__).resolve().parent.parent / "results/summaries"
    plots_dir = Path(__file__).resolve().parent.parent / "results/plots/budget_sweep"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = summaries_dir / "budget_sweep_results.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Calculate successful attack metric: constraint_pass == True and prediction_drift > 0.05
    df["is_success"] = (df["constraint_pass"] == True) & (df["prediction_drift"] > 0.05)
    
    # We want to aggregate over seeds, so we group by attack_name, query_budget, and seed first
    seed_agg = df.groupby(["attack_name", "query_budget", "seed"]).agg(
        mean_drift=("prediction_drift", "mean"),
        median_drift=("prediction_drift", "median"),
        success_rate=("is_success", "mean"),
        validity_rate=("valid_rdkit", "mean"),
        mean_runtime=("runtime_seconds", "mean"),
        queries_used=("query_count", "mean"),
    ).reset_index()

    # Then group by attack_name and query_budget to get mean and std across seeds
    final_agg = seed_agg.groupby(["attack_name", "query_budget"]).agg(
        drift_mean=("mean_drift", "mean"),
        drift_std=("mean_drift", "std"),
        median_drift_mean=("median_drift", "mean"),
        median_drift_std=("median_drift", "std"),
        success_mean=("success_rate", "mean"),
        success_std=("success_rate", "std"),
        validity_mean=("validity_rate", "mean"),
        validity_std=("validity_rate", "std"),
        runtime_mean=("mean_runtime", "mean"),
        runtime_std=("mean_runtime", "std"),
        queries_mean=("queries_used", "mean"),
        queries_std=("queries_used", "std")
    ).reset_index()

    attackers = final_agg["attack_name"].unique()
    
    def plot_metric(metric_mean, metric_std, title, ylabel, filename):
        plt.figure(figsize=(8, 6))
        for attack in attackers:
            subset = final_agg[final_agg["attack_name"] == attack]
            plt.errorbar(subset["query_budget"], subset[metric_mean], yerr=subset[metric_std], marker='o', label=attack, capsize=5)
        plt.title(title)
        plt.xlabel("Query Budget")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / filename)
        plt.close('all')

    plot_metric("drift_mean", "drift_std", "Mean Drift vs Query Budget", "Mean Drift", "mean_drift_vs_budget.png")
    plot_metric("median_drift_mean", "median_drift_std", "Median Drift vs Query Budget", "Median Drift", "median_drift_vs_budget.png")
    plot_metric("success_mean", "success_std", "Success Rate vs Query Budget", "Success Rate", "success_rate_vs_budget.png")
    plot_metric("validity_mean", "validity_std", "Validity Rate vs Query Budget", "RDKit Validity Rate", "validity_rate_vs_budget.png")
    plot_metric("runtime_mean", "runtime_std", "Runtime vs Query Budget", "Mean Runtime (s)", "runtime_vs_budget.png")
    plot_metric("queries_mean", "queries_std", "Queries Used vs Query Budget", "Mean Queries", "queries_vs_budget.png")

    # Similarity vs Drift (scatter of all runs)
    plt.figure(figsize=(8, 6))
    for attack in attackers:
        subset = df[df["attack_name"] == attack]
        plt.scatter(subset["tanimoto_similarity"], subset["prediction_drift"], label=attack, alpha=0.3, s=10)
    plt.title("Similarity vs Drift")
    plt.xlabel("Tanimoto Similarity")
    plt.ylabel("Prediction Drift")
    plt.legend()
    plt.savefig(plots_dir / "similarity_vs_drift.png")
    plt.close('all')

    print(f"Generated budget sweep plots in {plots_dir}")

if __name__ == "__main__":
    main()
