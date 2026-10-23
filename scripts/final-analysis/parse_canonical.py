import json
import pandas as pd

with open('materials-adversarial/results/canonical_benchmark_no_leakage.json') as f:
    data = json.load(f)

matrix = data['experiments_matrix']

# Phase 4 and 5 data
records = []
for m in matrix:
    records.append({
        'Experiment': m['experiment'],
        'Clean_RMSE': m['clean_rmse_eV'],
        'RandSMILES_Drift': m['rand_smiles_drift_mean_eV'],
        'MCMC_Drift': m['mcmc_drift_mean_eV']
    })

df = pd.DataFrame(records)
df.to_csv('results/final/defense_transfer.csv', index=False)
df.to_csv('results/final/ablation_results.csv', index=False)
