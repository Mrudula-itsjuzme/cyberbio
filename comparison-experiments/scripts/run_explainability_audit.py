import pandas as pd
import numpy as np
import json
import os

df = pd.read_csv("comparison-experiments/results/raw_explainability/mcmc_occlusion.csv")

# Verify constraints
sources = df["source_id"].unique()
assert len(sources) == 20, f"Expected 20 sources, got {len(sources)}"
assert "clean" in df["type"].unique(), "Clean rows missing"
assert "mcmc_adv" in df["type"].unique(), "MCMC rows missing"
assert not df["delta"].isnull().any(), "Empty attribution deltas"
assert (df["delta"] >= 0).all(), "Negative deltas not possible for abs()"

# Alignment via DP (Needleman-Wunsch for tokens)
def align_tokens(t1, t2):
    n, m = len(t1), len(t2)
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1): dp[i, 0] = i
    for j in range(m + 1): dp[0, j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if t1[i-1] == t2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j-1], dp[i-1, j], dp[i, j-1])
                
    # Backtrack to find aligned pairs and edit types
    i, j = n, m
    aligned = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and t1[i-1] == t2[j-1]:
            aligned.append((i-1, j-1, "matched"))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + 1:
            aligned.append((i-1, j-1, "substituted"))
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
            aligned.append((i-1, None, "deleted"))
            i -= 1
        else:
            aligned.append((None, j-1, "inserted"))
            j -= 1
            
    return aligned[::-1]

alignment_stats = []
summary = []

for sid in sources:
    c_df = df[(df["source_id"] == sid) & (df["type"] == "clean")].sort_values("position")
    a_df = df[(df["source_id"] == sid) & (df["type"] == "mcmc_adv")].sort_values("position")
    
    c_tokens = c_df["token"].tolist()
    a_tokens = a_df["token"].tolist()
    c_deltas = c_df["delta"].values
    a_deltas = a_df["delta"].values
    
    aligned = align_tokens(c_tokens, a_tokens)
    
    for c_idx, a_idx, op in aligned:
        c_val = c_deltas[c_idx] if c_idx is not None else None
        a_val = a_deltas[a_idx] if a_idx is not None else None
        alignment_stats.append({
            "source_id": sid, "op": op, "clean_pos": c_idx, "adv_pos": a_idx,
            "clean_delta": c_val, "adv_delta": a_val
        })
        
    # Top 10% positions in clean
    k = max(1, len(c_deltas) // 10)
    top_c_idx = np.argsort(c_deltas)[-k:]
    top_a_idx = np.argsort(a_deltas)[-k:]
    
    # Map top_c_idx to a_idx using alignment
    c_to_a = {c: a for c, a, op in aligned if c is not None and a is not None}
    mapped_top_c_in_a = [c_to_a.get(idx) for idx in top_c_idx if c_to_a.get(idx) is not None]
    overlap = len(set(mapped_top_c_in_a).intersection(set(top_a_idx)))
    
    # Are adversarial edits near high-attribution regions?
    # Get all indices in clean that were substituted or deleted
    modified_c_idx = [c for c, a, op in aligned if c is not None and op in ("substituted", "deleted")]
    if modified_c_idx:
        mean_attr_modified = np.mean(c_deltas[modified_c_idx])
    else:
        mean_attr_modified = 0.0
        
    summary.append({
        "source_id": sid,
        "clean_mean_attr": np.mean(c_deltas),
        "adv_mean_attr": np.mean(a_deltas),
        "clean_max_attr": np.max(c_deltas),
        "top_10_pct_overlap": overlap / k,
        "mean_attr_of_modified_tokens": mean_attr_modified,
        "mean_attr_overall": np.mean(c_deltas)
    })

os.makedirs("comparison-experiments/results/explainability", exist_ok=True)
pd.DataFrame(alignment_stats).to_csv("comparison-experiments/results/explainability/attribution_alignment.csv", index=False)
pd.DataFrame(summary).to_csv("comparison-experiments/results/explainability/mcmc_occlusion_summary.csv", index=False)

with open("comparison-experiments/results/explainability/attribution_statistics.json", "w") as f:
    json.dump({
        "mean_overlap": np.mean([s["top_10_pct_overlap"] for s in summary]),
        "mean_attr_modified": np.mean([s["mean_attr_of_modified_tokens"] for s in summary]),
        "mean_attr_overall": np.mean([s["mean_attr_overall"] for s in summary])
    }, f, indent=4)

print("Phase B Complete.")
