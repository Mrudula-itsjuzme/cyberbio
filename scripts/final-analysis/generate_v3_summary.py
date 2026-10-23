import pandas as pd
import scipy.stats as stats
import numpy as np

df = pd.read_csv("comparison-experiments/results/recomputed_metrics/classical_candidate_metrics.csv")

# Define success: constraint_pass (we use original or recomputed) and drift > 0.05
df['success_configured'] = (df['constraint_pass'] == True) & (df['prediction_drift'] > 0.05)

summary = []
for (attack, budget, seed), grp in df.groupby(['attack_family', 'query_budget', 'seed']):
    summary.append({
        'attack_family': attack,
        'budget': budget,
        'seed': seed,
        'n': len(grp),
        'mean_drift': grp['prediction_drift'].mean(),
        'median_drift': grp['prediction_drift'].median(),
        'p90_drift': grp['prediction_drift'].quantile(0.9),
        'std_drift': grp['prediction_drift'].std(),
        'max_drift': grp['prediction_drift'].max(),
        'configured_success_rate': grp['success_configured'].mean(),
        'raw_validity': grp['recomputed_valid_rdkit'].mean(),
        'constraint_pass_rate': grp['constraint_pass'].mean(),
        'mean_tanimoto': grp['recomputed_tanimoto'].mean(),
        'median_tanimoto': grp['recomputed_tanimoto'].median(),
        'mean_edit_distance': grp['recomputed_edit_distance'].mean(),
        'duplicate_rate': grp['duplicate_proposals'].mean() if 'duplicate_proposals' in grp.columns else 0.0,
        'mean_queries': grp['query_count'].mean(),
        'mean_runtime': grp['runtime_seconds'].mean()
    })

df_summary = pd.DataFrame(summary)
df_summary.to_csv("comparison-experiments/results/summaries/classical_attack_comparison_v3.csv", index=False)
df_summary.to_json("comparison-experiments/results/summaries/classical_attack_comparison_v3.json", orient='records', indent=4)
print("V3 Summary generated.")

# Paired stats
# Compare drift for Random vs MCMC and MCMC vs Evolutionary at budget 50
df_50 = df[df['query_budget'] == 50]
random_50 = df_50[df_50['attack_family'] == 'random'].groupby('source_id')['prediction_drift'].mean().sort_index()
mcmc_50 = df_50[df_50['attack_family'] == 'mcmc'].groupby('source_id')['prediction_drift'].mean().sort_index()
evo_50 = df_50[df_50['attack_family'] == 'evolutionary'].groupby('source_id')['prediction_drift'].mean().sort_index()

# Ensure matching sources
common = random_50.index.intersection(mcmc_50.index).intersection(evo_50.index)
random_50 = random_50.loc[common]
mcmc_50 = mcmc_50.loc[common]
evo_50 = evo_50.loc[common]

w_rm, p_rm = stats.wilcoxon(random_50, mcmc_50)
w_me, p_me = stats.wilcoxon(mcmc_50, evo_50)

stats_res = pd.DataFrame({
    'comparison': ['Random vs MCMC', 'MCMC vs Evolutionary'],
    'wilcoxon_stat': [w_rm, w_me],
    'p_value': [p_rm, p_me]
})
import os
os.makedirs("comparison-experiments/results/statistics", exist_ok=True)
stats_res.to_csv("comparison-experiments/results/statistics/classical_paired_stats.csv", index=False)
print("Stats generated.")
