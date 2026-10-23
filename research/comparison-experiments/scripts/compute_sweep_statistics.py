import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def bootstrap_ci(data, stat_func=np.mean, n_bootstraps=1000, ci=95):
    bootstrapped_stats = []
    n = len(data)
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_stats.append(stat_func(sample))
    lower = np.percentile(bootstrapped_stats, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_stats, 100 - (100 - ci) / 2)
    return lower, upper

def main():
    summaries_dir = Path(__file__).resolve().parent.parent / "results/summaries"
    csv_path = summaries_dir / "budget_sweep_results.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    with open(summaries_dir / "sweep_statistical_analysis.txt", "w") as f:
        f.write("=== BUDGET SWEEP STATISTICAL ANALYSIS ===\n\n")
        
        for budget in df["query_budget"].unique():
            f.write(f"--- Budget: {budget} ---\n")
            df_budget = df[df["query_budget"] == budget]
            
            attackers = df_budget["attack_name"].unique()
            
            for attacker in attackers:
                df_attack = df_budget[df_budget["attack_name"] == attacker]
                drifts = df_attack["prediction_drift"].values
                mean_drift = np.mean(drifts)
                std_drift = np.std(drifts)
                
                lower, upper = bootstrap_ci(drifts, np.mean)
                f.write(f"Attacker: {attacker}\n")
                f.write(f"  Mean Drift: {mean_drift:.4f} (std: {std_drift:.4f})\n")
                f.write(f"  95% CI (Bootstrap): [{lower:.4f}, {upper:.4f}]\n\n")
            
            # Paired deltas (Random vs MCMC, etc.)
            # Need to align by source_id and seed
            f.write("Paired Tests (Wilcoxon Signed-Rank):\n")
            for i in range(len(attackers)):
                for j in range(i+1, len(attackers)):
                    att1 = attackers[i]
                    att2 = attackers[j]
                    
                    df1 = df_budget[df_budget["attack_name"] == att1].set_index(["source_id", "seed"])
                    df2 = df_budget[df_budget["attack_name"] == att2].set_index(["source_id", "seed"])
                    
                    common_idx = df1.index.intersection(df2.index)
                    if len(common_idx) > 0:
                        drift1 = df1.loc[common_idx, "prediction_drift"].values
                        drift2 = df2.loc[common_idx, "prediction_drift"].values
                        
                        delta = drift1 - drift2
                        mean_delta = np.mean(delta)
                        
                        try:
                            stat, pval = wilcoxon(drift1, drift2)
                            f.write(f"  {att1} vs {att2}:\n")
                            f.write(f"    Mean Paired Delta ({att1} - {att2}): {mean_delta:.4f}\n")
                            f.write(f"    Wilcoxon p-value: {pval:.4e}\n")
                        except ValueError:
                            # if all differences are 0
                            f.write(f"  {att1} vs {att2}: All differences are zero.\n")
            f.write("\n")

    print(f"Generated statistical analysis at {summaries_dir / 'sweep_statistical_analysis.txt'}")

if __name__ == "__main__":
    main()
