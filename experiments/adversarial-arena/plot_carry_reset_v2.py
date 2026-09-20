import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results/final/carry_reset", exist_ok=True)
df = pd.read_csv("results/final/carry_reset/carry_reset_events_v2.csv")

# 1. gcarry_mean_matched_by_round.png
matched = df[df["query_match_status"] == "QUERY_MATCHED"]
plt.figure(figsize=(10, 6))
sns.lineplot(data=matched, x="round", y="G_carry_mean", hue="method", marker='o')
plt.title("G_carry_mean by Round (Matched Queries Only)")
plt.axhline(0, color='black', linestyle='--')
plt.savefig("results/final/carry_reset/gcarry_mean_matched_by_round.png")
plt.close()

# 2. gcarry_best_matched_by_round.png
plt.figure(figsize=(10, 6))
sns.lineplot(data=matched, x="round", y="G_carry_best", hue="method", marker='o')
plt.title("G_carry_best by Round (Matched Queries Only)")
plt.axhline(0, color='black', linestyle='--')
plt.savefig("results/final/carry_reset/gcarry_best_matched_by_round.png")
plt.close()

# 3. gcarry_all_events_by_round.png
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="round", y="G_carry_mean", hue="method", style="query_match_status", markers=True, errorbar=None)
plt.title("G_carry_mean by Round (All Events)")
plt.axhline(0, color='black', linestyle='--')
plt.savefig("results/final/carry_reset/gcarry_all_events_by_round.png")
plt.close()

# 4. query_mismatch_by_method_round.png
plt.figure(figsize=(10, 6))
mismatched = df[df["query_match_status"] == "QUERY_MISMATCHED"]
sns.barplot(data=mismatched, x="round", y="delta_queries", hue="method")
plt.title("Query Overshoot/Undershoot (Carry - Reset)")
plt.axhline(0, color='black')
plt.savefig("results/final/carry_reset/query_mismatch_by_method_round.png")
plt.close()

# 5. drift_per_query_by_round.png
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="round", y="carry_mean_drift_per_query", hue="method", marker='o')
sns.lineplot(data=df, x="round", y="reset_mean_drift_per_query", hue="method", style=True, dashes=[(2, 2)], marker='s')
plt.title("Mean Drift per Query by Round (Secondary Analysis)")
plt.savefig("results/final/carry_reset/drift_per_query_by_round.png")
plt.close()

# 6. per_seed_postinit_gcarry.png
plt.figure(figsize=(12, 6))
postinit = df[df["round"] > 0]
sns.barplot(data=postinit, x="seed", y="G_carry_mean", hue="method", errorbar=None)
plt.title("G_carry_mean Post-Init by Seed (All Events)")
plt.savefig("results/final/carry_reset/per_seed_postinit_gcarry.png")
plt.close()

# 7-9. Trajectories
for m in ["cem", "smc", "validity_smc"]:
    plt.figure(figsize=(10, 6))
    sub = df[df["method"] == m]
    if len(sub) > 0:
        sns.lineplot(data=sub, x="round", y="carry_mean_drift", style="query_match_status", markers=True, )
        sns.lineplot(data=sub, x="round", y="reset_mean_drift", style="query_match_status", markers=True, dashes=[(2,2)])
        plt.title(f"{m.upper()} Memory Effect")
        plt.savefig(f"results/final/carry_reset/{m}_memory_effect.png")
    plt.close()

# 10. smc_vs_validity_smc_equivalence.png
plt.figure(figsize=(8, 8))
sub = df[df["method"].isin(["smc", "validity_smc"])]
sns.scatterplot(data=sub, x="round", y="G_carry_mean", hue="method", s=100, alpha=0.5)
plt.title("SMC vs Validity-SMC Equivalence")
plt.savefig("results/final/carry_reset/smc_vs_validity_smc_equivalence.png")
plt.close()
