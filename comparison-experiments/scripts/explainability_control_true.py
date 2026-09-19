import pandas as pd
import numpy as np

# Load alignment results
df_align = pd.read_csv("results/explainability/attribution_alignment.csv")
df_raw = pd.read_csv("results/raw_explainability/mcmc_occlusion.csv")

# Ensure df_align has the right columns (clean_delta)
sources = df_align["source_id"].unique()
results = []
diffs = []
np.random.seed(42)

for src in sources:
    align_src = df_align[df_align["source_id"] == src]
    raw_src = df_raw[df_raw["source_id"] == src]
    
    # Edited positions
    edited = align_src[align_src["op"] != "matched"]
    if len(edited) == 0:
        continue
        
    edited_attrs = edited["clean_delta"].abs().values
    mean_edited = np.mean(edited_attrs)
    
    # Unedited positions (from raw)
    # The raw attribution array has attributions for each position.
    # We sample `len(edited)` positions 1000 times.
    all_attrs = raw_src["delta"].abs().values
    
    means_random = []
    for _ in range(1000):
        # sample without replacement if possible, or with replacement
        if len(all_attrs) >= len(edited):
            samp = np.random.choice(all_attrs, size=len(edited), replace=False)
        else:
            samp = np.random.choice(all_attrs, size=len(edited), replace=True)
        means_random.append(np.mean(samp))
        
    mean_random = np.mean(means_random)
    results.append({
        "source_id": src,
        "mean_edited": mean_edited,
        "mean_matched_random": mean_random,
        "paired_diff": mean_edited - mean_random
    })
    diffs.append(mean_edited - mean_random)

if len(results) > 0:
    res_df = pd.DataFrame(results)
    n_sources = len(res_df)
    mean_edited_attribution = res_df["mean_edited"].mean()
    mean_matched_random_attribution = res_df["mean_matched_random"].mean()
    paired_difference = res_df["paired_diff"].mean()
    
    # Bootstrap 95% CI
    boot_diffs = [np.random.choice(diffs, len(diffs), replace=True).mean() for _ in range(1000)]
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    
    # Empirical p-value
    # How many times is random mean >= edited mean across all sources? 
    # Or test if diffs is significantly > 0
    from scipy.stats import wilcoxon
    if len(diffs) >= 10:
        w_stat, p_val = wilcoxon(diffs)
    else:
        p_val = 1.0 # fallback
        
    # Effect size (Cohen's d)
    effect_size = paired_difference / np.std(diffs, ddof=1) if len(diffs) > 1 else 0
    
    with open("results/explainability/true_random_control.txt", "w") as f:
        f.write(f"n_sources: {n_sources}\n")
        f.write(f"mean_edited_attribution: {mean_edited_attribution}\n")
        f.write(f"mean_matched_random_attribution: {mean_matched_random_attribution}\n")
        f.write(f"paired_difference: {paired_difference}\n")
        f.write(f"bootstrap_95CI: [{ci_low}, {ci_high}]\n")
        f.write(f"empirical_p_value: {p_val}\n")
        f.write(f"effect_size: {effect_size}\n")
else:
    print("No edited positions found.")
