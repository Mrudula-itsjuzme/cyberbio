import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

os.makedirs("results/final/phase2_plots", exist_ok=True)
audit_log = open("docs/ARENA_PHASE2_FIGURE_AUDIT.md", "w")
audit_log.write("# Phase 2 Figure Audit\n\n")

def plot_fig(filename, csv_file, x_col, y_col, group_by=None, title="", hlines=None, plot_type="line"):
    csv_path = f"results/final/phase2/{csv_file}"
    img_path = f"results/final/phase2_plots/{filename}"
    if not os.path.exists(csv_path):
        audit_log.write(f"- {filename} NOT generated (missing {csv_file})\n")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        audit_log.write(f"- {filename} NOT generated (missing columns {x_col},{y_col} in {csv_file})\n")
        return
        
    plt.figure(figsize=(8,6))
    
    if plot_type == "line":
        if group_by and group_by in df.columns:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=group_by, marker="o")
        else:
            sns.lineplot(data=df, x=x_col, y=y_col, marker="o")
    elif plot_type == "scatter":
        if group_by and group_by in df.columns:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=group_by)
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col)
    elif plot_type == "bar":
        if group_by and group_by in df.columns:
            sns.barplot(data=df, x=x_col, y=y_col, hue=group_by)
        else:
            sns.barplot(data=df, x=x_col, y=y_col)
            
    if hlines:
        for val in hlines:
            plt.axhline(val, color="red", linestyle="--", alpha=0.5)
            
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    
    audit_log.write(f"- {filename} | {csv_file} | x={x_col} | y={y_col} | group={group_by}\n")

plot_fig("phase2_drift_vs_round.png", "phase2_attackers.csv", "round", "mean_drift", "method", "Mean Drift vs Round")
plot_fig("phase2_g_adapt.png", "phase2_g_adapt.csv", "round", "G_adapt", "method", "G_adapt", hlines=[0])
plot_fig("phase2_defender_gain.png", "phase2_defenders.csv", "round", "defender_gain_difference", "method", "Defender Gain Difference")
plot_fig("phase2_heldout_transfer.png", "phase2_transfer.csv", "round", "T_Evo", "method", "Held-out Transfer (Evolutionary)", hlines=[0])

# Redistribution
df_r = pd.read_csv("results/final/phase2/phase2_redistribution.csv")
if not df_r.empty:
    r_counts = df_r.groupby("method")["redistribution_observed"].sum().reset_index()
    r_counts.to_csv("results/final/phase2/phase2_redistribution_counts.csv", index=False)
    plot_fig("phase2_redistribution.png", "phase2_redistribution_counts.csv", "method", "redistribution_observed", title="Redistribution Events by Method", plot_type="bar")
else:
    audit_log.write("- phase2_redistribution.png NOT generated (empty data)\n")

# Candidate Overlap
plot_fig("phase2_candidate_overlap.png", "phase2_attacker_overlap.csv", "round", "jaccard_overlap", "method_a", "Candidate Overlap (Jaccard)", plot_type="scatter")

plot_fig("cem_normalized_entropy.png", "phase2_cem_collapse.csv", "round", "normalized_entropy", "seed", "CEM Normalized Entropy", hlines=[0.20, 0.50])
plot_fig("smc_ess_ratio.png", "phase2_smc_degeneracy.csv", "round", "ess_ratio", "method", "SMC ESS Ratio", hlines=[0.20, 0.50])
plot_fig("validity_smc_efficiency.png", "phase2_validity_efficiency.csv", "round", "valid_per_target_query", "method", "Valid Proposals per Target Query")
plot_fig("clean_robust_tradeoff_phase2.png", "phase2_clean_robust_tradeoff.csv", "delta_clean", "delta_robust", "classification", "Clean vs Robust Tradeoff", plot_type="scatter")
plot_fig("phase2_method_summary.png", "phase2_summary.csv", "method", "mean_drift", title="Mean Drift Summary by Method", plot_type="bar")

# per seed trajectory
plot_fig("per_seed_phase2_trajectories.png", "phase2_attackers.csv", "round", "mean_drift", "seed", "Per-seed Mean Drift Trajectories")

audit_log.close()
print("Plotting complete")
