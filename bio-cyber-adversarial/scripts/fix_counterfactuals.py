import pandas as pd
import numpy as np
import json

df = pd.read_csv("results/v3/relationship_counterfactuals.csv")
orig = df["orig_prob"] if "orig_prob" in df.columns else df["original_probability"]
cols = [c for c in df.columns if c not in ["orig_prob", "original_probability"]]

summary = {}
for c in cols:
    diffs = (df[c] - orig).abs().values
    means = [np.mean(np.random.choice(diffs, len(diffs), replace=True)) for _ in range(1000)]
    summary[c] = {
        "mean_abs_change": float(np.mean(diffs)),
        "bootstrap_95_CI": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "fraction_crossing_threshold": float(((df[c] < 0.5) & (orig >= 0.5)).mean())
    }

with open("results/v3/counterfactual_summary.json", "w") as f:
    json.dump(summary, f, indent=4)
