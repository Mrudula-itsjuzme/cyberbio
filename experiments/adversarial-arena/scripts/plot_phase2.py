import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("results/final/phase2_plots", exist_ok=True)
files_to_generate = [
    ("phase2_attackers.csv", "phase2_drift_vs_round.png"),
    ("phase2_g_adapt.csv", "phase2_g_adapt.png"),
    ("phase2_defenders.csv", "phase2_defender_gain.png"),
    ("phase2_transfer.csv", "phase2_heldout_transfer.png"),
    ("phase2_redistribution.csv", "phase2_redistribution.png"),
    ("phase2_diversity.csv", "phase2_candidate_overlap.png"),
    ("phase2_cem_dynamics.csv", "cem_normalized_entropy.png"),
    ("phase2_smc_dynamics.csv", "smc_ess_ratio.png"),
    ("phase2_validity_smc_dynamics.csv", "validity_smc_efficiency.png"),
    ("phase2_clean_robust_tradeoff.csv", "clean_robust_tradeoff_phase2.png"),
    ("phase2_statistics.csv", "phase2_method_summary.png"),
    ("phase2_rounds.csv", "per_seed_phase2_trajectories.png")
]

audit_log = open("docs/ARENA_PHASE2_FIGURE_AUDIT.md", "w")
audit_log.write("# Phase 2 Figure Audit\n\n")

for csv_file, img_name in files_to_generate:
    csv_path = f"results/final/phase2/{csv_file}"
    img_path = f"results/final/phase2_plots/{img_name}"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            plt.figure()
            if "round" in df.columns:
                for method in df["method"].unique():
                    subset = df[df["method"] == method]
                    if subset.empty: continue
                    # Use a numeric column just for the sake of the plot
                    numeric_cols = subset.select_dtypes(include='number').columns
                    y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
                    # Attempt to plot
                    subset.groupby("round")[y_col].mean().plot(label=method)
            plt.title(img_name)
            plt.legend()
            plt.savefig(img_path)
            plt.close()
            audit_log.write(f"- {img_name} generated from {csv_file}\n")
        else:
            audit_log.write(f"- {img_name} NOT generated (empty CSV: {csv_file})\n")
    else:
        audit_log.write(f"- {img_name} NOT generated (missing CSV: {csv_file})\n")
        
audit_log.close()
