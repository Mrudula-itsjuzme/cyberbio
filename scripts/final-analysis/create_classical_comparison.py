import pandas as pd
import json

df = pd.read_csv("comparison-experiments/results/summaries/budget_sweep_results.csv")

def aggregate(df):
    results = []
    for (family, budget, seed), group in df.groupby(['attack_family', 'query_budget', 'seed']):
        r = {
            'attack_family': family,
            'budget': budget,
            'seed': seed,
            'mean_drift': group['prediction_drift'].mean(),
            'median_drift': group['prediction_drift'].median(),
            'p90_drift': group['prediction_drift'].quantile(0.9),
            'max_drift': group['prediction_drift'].max(),
            'success_rate': (group['prediction_drift'] > 0).mean(), # assuming success is drift > 0
            'raw_proposal_validity': group['valid_rdkit'].mean(),
            'constraint_pass_rate': group['constraint_pass'].mean(),
            'mean_tanimoto': group['tanimoto_similarity'].mean(),
            'mean_edit_distance': group['edit_distance'].mean(),
            'query_count_mean': group['query_count'].mean(),
            'runtime_mean': group['runtime_seconds'].mean()
        }
        results.append(r)
    return pd.DataFrame(results)

agg_df = aggregate(df)
agg_df.to_csv("comparison-experiments/results/summaries/classical_attack_comparison.csv", index=False)
agg_df.to_json("comparison-experiments/results/summaries/classical_attack_comparison.json", orient="records", indent=4)
print("Saved classical comparison.")
