import json, os, math
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("../results/materials/per_example_results.csv")
budgets = [5, 20, 50]
attacks = sorted(df["attack_condition"].unique())

agg_rows = []
for atk in attacks:
    for b in budgets:
        sub = df[(df["attack_condition"] == atk) & (df["search_budget"] == b)]
        if sub.empty:
            continue
        n = len(sub)
        prop = sub["proposal_attempts"].replace(0, np.nan)
        row = dict(domain="materials", attack_condition=atk, search_budget=b, n=n,
            mean_absolute_drift=sub["absolute_prediction_change"].mean(),
            median_absolute_drift=sub["absolute_prediction_change"].median(),
            p90_absolute_drift=sub["absolute_prediction_change"].quantile(0.9),
            mean_edit_distance=sub["edit_distance"].mean(),
            median_edit_distance=sub["edit_distance"].median(),
            mean_tanimoto_similarity=sub["tanimoto_similarity"].mean(),
            mean_proposal_attempts=sub["proposal_attempts"].mean(),
            mean_valid_candidates=sub["valid_candidates"].mean(),
            mean_attack_model_queries=sub["attack_model_queries"].mean(),
            mean_attribution_queries=sub["attribution_queries"].mean(),
            mean_total_model_queries=sub["total_model_queries"].mean(),
            rdkit_proposal_validity_rate=(sub["rdkit_valid_proposals"] / prop).mean(),
            canonicalization_proposal_validity_rate=(sub["canonicalizable_proposals"] / prop).mean(),
            tokenization_proposal_validity_rate=(sub["tokenizable_proposals"] / prop).mean(),
            selected_rdkit_valid_rate=sub["selected_candidate_rdkit_valid"].mean(),
            selected_canonicalization_valid_rate=sub["selected_candidate_canonicalization_valid"].mean(),
            selected_tokenization_valid_rate=sub["selected_candidate_tokenization_valid"].mean(),
            frac_query_budget_exhausted=(sub["termination_reason"] == "QUERY_BUDGET_EXHAUSTED").mean(),
            frac_proposal_cap_reached=(sub["termination_reason"] == "PROPOSAL_CAP_REACHED").mean(),
            frac_no_valid_candidate=(sub["termination_reason"] == "NO_VALID_CANDIDATE").mean(),
            mean_runtime=sub["runtime_seconds"].mean(),
            median_runtime=sub["runtime_seconds"].median())
        agg_rows.append(row)

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv("../results/materials/aggregated_results.csv", index=False)

# Proposal-validity summary
validity_rows = []
for atk in attacks:
    for b in budgets:
        sub = df[(df["attack_condition"] == atk) & (df["search_budget"] == b)]
        if sub.empty: continue
        tp = sub["proposal_attempts"].sum()
        validity_rows.append(dict(
            attack_condition=atk, search_budget=b,
            total_proposal_attempts=tp,
            rdkit_valid_proposals=sub["rdkit_valid_proposals"].sum(),
            canonicalizable_proposals=sub["canonicalizable_proposals"].sum(),
            tokenizable_proposals=sub["tokenizable_proposals"].sum(),
            rdkit_proposal_validity_rate=sub["rdkit_valid_proposals"].sum()/tp if tp else np.nan,
            canonicalization_proposal_validity_rate=sub["canonicalizable_proposals"].sum()/tp if tp else np.nan,
            tokenization_proposal_validity_rate=sub["tokenizable_proposals"].sum()/tp if tp else np.nan,
        ))
pd.DataFrame(validity_rows).to_csv("../results/materials/proposal_validity_summary.csv", index=False)

# Attribution overhead
attr_sub = df[df["attack_condition"] == "Attribution-High"]
attr_summary = dict(
    mean=float(attr_sub["attribution_queries"].mean()),
    median=float(attr_sub["attribution_queries"].median()),
    min=int(attr_sub["attribution_queries"].min()),
    max=int(attr_sub["attribution_queries"].max()),
    note="attribution_queries = len(tokenize(source)), NOT 256 padding length")

# Bootstrap CI + Wilcoxon
np.random.seed(42)
N_BOOT = 1000

def bci(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2: return [None, None]
    boot = [float(np.mean(np.random.choice(arr, len(arr), replace=True))) for _ in range(N_BOOT)]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

def cdz(arr):
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2 or np.std(arr, ddof=1) == 0: return None
    return float(np.mean(arr) / np.std(arr, ddof=1))

def ps(d1, d2, m):
    v1, v2 = d1[m].values, d2[m].values
    diffs = v1 - v2; dc = diffs[~np.isnan(diffs)]
    if len(dc) < 3: return {"n":len(dc),"mean_diff":None,"cohen_dz":None,"ci_lower":None,"ci_upper":None,"p_value":None}
    try: p = float(wilcoxon(v1[~np.isnan(diffs)], v2[~np.isnan(diffs)]).pvalue)
    except: p = None
    ci = bci(dc)
    return {"n":len(dc),"mean_diff":float(np.mean(dc)),"cohen_dz":cdz(dc),"ci_lower":ci[0],"ci_upper":ci[1],"p_value":p}

comparisons = [("MCMC","Random"),("Evolutionary","Random"),
               ("Attribution-High","Attribution-Random"),("Attribution-Low","Attribution-Random")]
metrics = ["absolute_prediction_change","attack_model_queries","total_model_queries",
           "proposal_attempts","edit_distance","runtime_seconds"]

stats_out = []
for b in budgets:
    sub = df[df["search_budget"]==b]
    for c1,c2 in comparisons:
        d1 = sub[sub["attack_condition"]==c1].sort_values("source_id").reset_index(drop=True)
        d2 = sub[sub["attack_condition"]==c2].sort_values("source_id").reset_index(drop=True)
        if d1.empty or d2.empty: continue
        entry = {"comparison":f"{c1} vs {c2}","budget":b,"metrics":{}}
        for m in metrics:
            if m in d1.columns and m in d2.columns:
                entry["metrics"][m] = ps(d1,d2,m)
        stats_out.append(entry)

with open("../results/materials/statistical_results.json","w") as f:
    json.dump({"description":"Paired Wilcoxon + Bootstrap CI (N=1000, seed=42). No McNemar: success=NaN.",
               "attribution_overhead":attr_summary,"paired_comparisons":stats_out}, f, indent=2)

with open("../results/materials/bootstrap_metadata.json","w") as f:
    json.dump({"seed":42,"iterations":N_BOOT,"method":"percentile","phase":"4.1"}, f, indent=2)

print("Aggregation complete.")
print(agg_df[["attack_condition","search_budget","mean_absolute_drift",
              "mean_proposal_attempts","mean_attack_model_queries",
              "rdkit_proposal_validity_rate"]].to_string(index=False))
