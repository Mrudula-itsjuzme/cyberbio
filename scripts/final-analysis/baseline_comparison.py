import pandas as pd
import json

df_simple = pd.read_csv('materials-adversarial/results/phase6_attribution/run_01/simple_baseline_metrics.csv')
with open('materials-adversarial/results/canonical_benchmark_no_leakage.json') as f:
    canon = json.load(f)
    
transformer_r2 = [x['clean_r2'] for x in canon['experiments_matrix'] if x['experiment'] == 'RandomSplit_Baseline'][0]
transformer_rmse = [x['clean_rmse_eV'] for x in canon['experiments_matrix'] if x['experiment'] == 'RandomSplit_Baseline'][0]
transformer_mae = [x['clean_mae_eV'] for x in canon['experiments_matrix'] if x['experiment'] == 'RandomSplit_Baseline'][0]

df_trans = pd.DataFrame([{
    'model': 'TransPolymer',
    'mae': transformer_mae,
    'rmse': transformer_rmse,
    'r2': transformer_r2
}])

df_all = pd.concat([df_simple, df_trans], ignore_index=True)
df_all.to_csv('results/final/baseline_comparison.csv', index=False)
