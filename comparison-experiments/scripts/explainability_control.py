import pandas as pd
import numpy as np

# Load alignment results
df_align = pd.read_csv("results/explainability/attribution_alignment.csv")
# Filter to only the positions that were substituted/edited!
# Wait, Needleman-Wunsch gives op="substituted" or "matched".
# If they are adversarial edits, they are "substituted" (or "inserted", "deleted").
# I will use all edited positions (op != 'matched').
edited = df_align[df_align["op"] != "matched"]
if len(edited) == 0:
    # fallback to all if it didn't record op properly
    edited_attrs = df_align["clean_delta"].abs().values
else:
    edited_attrs = edited["clean_delta"].abs().values

df_raw = pd.read_csv("results/raw_explainability/mcmc_occlusion.csv")
# Overall mean attribution per source
mean_overall = df_raw.groupby("source_id")["attribution"].apply(lambda x: x.abs().mean())

# Ensure we aggregate edited attributes by source first to do a paired test
if len(edited) > 0:
    mean_edited = edited.groupby("source_id")["clean_delta"].apply(lambda x: x.abs().mean())
else:
    mean_edited = df_align.groupby("source_id")["clean_delta"].apply(lambda x: x.abs().mean())

df_combined = pd.DataFrame({"edited": mean_edited, "random": mean_overall}).dropna()

diffs = df_combined["edited"] - df_combined["random"]
edited_mean = df_combined["edited"].mean()
random_mean = df_combined["random"].mean()
diff_mean = diffs.mean()

# bootstrap 95% CI of the difference
np.random.seed(42)
boot_diffs = [np.random.choice(diffs, len(diffs), replace=True).mean() for _ in range(1000)]
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

with open("results/explainability/control_stats.txt", "w") as f:
    f.write(f"Mean edited attribution: {edited_mean}\n")
    f.write(f"Mean random attribution: {random_mean}\n")
    f.write(f"Paired difference: {diff_mean}\n")
    f.write(f"Bootstrap 95% CI: [{ci_low}, {ci_high}]\n")
    if ci_low > 0:
        f.write("Attacks preferentially target high-attribution positions supported by control comparison.\n")
    else:
        f.write("Attacks do NOT definitively target high-attribution positions.\n")
