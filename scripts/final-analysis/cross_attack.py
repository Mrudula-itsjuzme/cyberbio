import pandas as pd
import json
import matplotlib.pyplot as plt
import os

os.makedirs('results/final/figures', exist_ok=True)

with open('materials-adversarial/results/canonical_benchmark_no_leakage.json') as f:
    canon = json.load(f)

matrix = canon['experiments_matrix']
records = []
for m in matrix:
    if "RandomSplit" in m['experiment']:
        defense = m['experiment'].split('_')[1]
        records.append({
            "source_file": "canonical_benchmark_no_leakage.json",
            "experiment": m['experiment'],
            "split": "RandomSplit",
            "seed_aggregation": "single_seed_42",
            "defense": defense,
            "attack": "RandomSMILES",
            "metric": "mean_drift_eV",
            "value": m['rand_smiles_drift_mean_eV']
        })
        records.append({
            "source_file": "canonical_benchmark_no_leakage.json",
            "experiment": m['experiment'],
            "split": "RandomSplit",
            "seed_aggregation": "single_seed_42",
            "defense": defense,
            "attack": "MCMC",
            "metric": "mean_drift_eV",
            "value": m['mcmc_drift_mean_eV']
        })

df = pd.DataFrame(records)
df.to_csv('results/final/cross_attack_generalization.csv', index=False)

# create heatmap data
heatmap_data = df.pivot(index='defense', columns='attack', values='value')
# Reorder index to match semantic progression
heatmap_data = heatmap_data.reindex(['Baseline', 'MCMCDefended', 'RandSmilesAug', 'CombinedDefense'])

plt.figure(figsize=(6,4))
plt.imshow(heatmap_data.values, cmap='viridis')
plt.colorbar(label='Mean Prediction Drift (eV)')
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
plt.title('Cross-Attack Generalization Matrix (RandomSplit)')
plt.tight_layout()
plt.savefig('results/final/figures/cross_attack_heatmap.png')
plt.close()
