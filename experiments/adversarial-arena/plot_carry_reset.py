import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results/final/carry_reset", exist_ok=True)

df = pd.read_csv("results/final/carry_reset/carry_reset_events.csv")
multiseed = pd.read_csv("results/final/carry_reset/carry_reset_multiseed_statistics.csv")

# 1. gcarry_mean_by_round.png
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="round", y="G_carry_mean", hue="method", errorbar='sd', marker='o')
plt.title("G_carry_mean by Round across Seeds")
plt.axhline(0, color='black', linestyle='--')
plt.savefig("results/final/carry_reset/gcarry_mean_by_round.png")
plt.close()

# 2. gcarry_best_by_round.png
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="round", y="G_carry_best", hue="method", errorbar='sd', marker='o')
plt.title("G_carry_best by Round across Seeds")
plt.axhline(0, color='black', linestyle='--')
plt.savefig("results/final/carry_reset/gcarry_best_by_round.png")
plt.close()

# 3. carry_vs_reset_mean_drift.png
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df, x="reset_mean_drift", y="carry_mean_drift", hue="method", style="round")
plt.plot([0, df["reset_mean_drift"].max()], [0, df["reset_mean_drift"].max()], 'k--')
plt.title("Carry vs Reset Mean Drift (Points above line = Carry Advantage)")
plt.savefig("results/final/carry_reset/carry_vs_reset_mean_drift.png")
plt.close()

# 4. carry_vs_reset_best_drift.png
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df, x="reset_best_drift", y="carry_best_drift", hue="method", style="round")
plt.plot([0, df["reset_best_drift"].max()], [0, df["reset_best_drift"].max()], 'k--')
plt.title("Carry vs Reset Best Drift")
plt.savefig("results/final/carry_reset/carry_vs_reset_best_drift.png")
plt.close()

# 5. carry_vs_reset_query_efficiency.png
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="round", y="carry_mean_drift_per_query", hue="method", label="Carry", marker='o')
sns.lineplot(data=df, x="round", y="reset_mean_drift_per_query", hue="method", style=True, dashes=[(2, 2)], label="Reset", marker='s')
plt.title("Query Efficiency: Mean Drift per Query by Round")
plt.savefig("results/final/carry_reset/carry_vs_reset_query_efficiency.png")
plt.close()

# 6. per_seed_gcarry.png
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="seed", y="G_carry_mean", hue="method", errorbar=None)
plt.title("G_carry_mean by Seed")
plt.savefig("results/final/carry_reset/per_seed_gcarry.png")
plt.close()

# Trajectories
for m in ["cem", "smc", "validity_smc"]:
    plt.figure(figsize=(10, 6))
    sub = df[df["method"] == m]
    if len(sub) > 0:
        sns.lineplot(data=sub, x="round", y="carry_mean_drift", label="Carry Mean", marker='o')
        sns.lineplot(data=sub, x="round", y="reset_mean_drift", label="Reset Mean", marker='s')
        plt.title(f"{m.upper()} Memory Trajectory")
        plt.savefig(f"results/final/carry_reset/{m}_memory_trajectory.png")
    plt.close()

# 10. gcarry_method_summary.png
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="method", y="G_carry_mean")
plt.axhline(0, color='black', linestyle='--')
plt.title("Overall G_carry_mean Summary by Method")
plt.savefig("results/final/carry_reset/gcarry_method_summary.png")
plt.close()
