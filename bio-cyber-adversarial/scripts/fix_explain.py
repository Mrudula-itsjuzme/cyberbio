import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import json

df = pd.read_csv("results/v3/explainability_raw.csv")
diffs = df["edited_attribution"] - df["random_attribution"]
means = [np.mean(np.random.choice(diffs, len(diffs), replace=True)) for _ in range(1000)]

res = wilcoxon(df["edited_attribution"], df["random_attribution"])

summary = {
    "n_sources": len(df),
    "mean_edited_attribution": float(df["edited_attribution"].mean()),
    "mean_random_attribution": float(df["random_attribution"].mean()),
    "paired_difference": float(diffs.mean()),
    "bootstrap_95_CI": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    "paired_statistical_test": "Wilcoxon",
    "p_value": float(res.pvalue),
    "effect_size": float(diffs.mean() / df["edited_attribution"].std())
}
with open("results/v3/explainability_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
