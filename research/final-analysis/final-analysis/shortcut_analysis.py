import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

os.makedirs('results/final/figures', exist_ok=True)

# Load dataset
df_orig = pd.read_csv('materials-adversarial/data/processed/processed.csv')
rep_col = 'original_representation'
if rep_col not in df_orig.columns: rep_col = 'smiles'

df_orig['seq_len'] = df_orig[rep_col].apply(len)

# Load predictions/attack records
records = []
with open('results/materials/experiments/phase2_paired_benchmark/clean_attack_records.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['validity_status'] == 'valid':
            records.append(r)

df_attacks = pd.DataFrame(records)
# Merge with df_orig to get true target
df_attacks = df_attacks.merge(df_orig[['polymer_id', 'property_value', 'seq_len']], left_on='sample_id', right_on='polymer_id')

df_attacks['adv_seq_len'] = df_attacks['adversarial_representation'].apply(len)
df_attacks['length_change'] = (df_attacks['adv_seq_len'] - df_attacks['seq_len']).abs()

df_baseline = df_attacks[['sample_id', 'seq_len', 'property_value', 'original_prediction']].drop_duplicates()
df_baseline['absolute_error'] = (df_baseline['property_value'] - df_baseline['original_prediction']).abs()

# A. correlation(sequence_length, true_target)
r_A, p_A = stats.pearsonr(df_baseline['seq_len'], df_baseline['property_value'])
rho_A, _ = stats.spearmanr(df_baseline['seq_len'], df_baseline['property_value'])

# B. correlation(sequence_length, model_prediction)
r_B, p_B = stats.pearsonr(df_baseline['seq_len'], df_baseline['original_prediction'])
rho_B, _ = stats.spearmanr(df_baseline['seq_len'], df_baseline['original_prediction'])

# C. correlation(sequence_length, absolute_prediction_error)
r_C, p_C = stats.pearsonr(df_baseline['seq_len'], df_baseline['absolute_error'])
rho_C, _ = stats.spearmanr(df_baseline['seq_len'], df_baseline['absolute_error'])

# D. correlation(abs(length_change), adversarial_prediction_drift)
r_D, p_D = stats.pearsonr(df_attacks['length_change'], df_attacks['absolute_prediction_drift'])
rho_D, _ = stats.spearmanr(df_attacks['length_change'], df_attacks['absolute_prediction_drift'])

stats_data = [
    {"comparison": "length vs true_target", "n": len(df_baseline), "pearson_r": r_A, "pearson_p": p_A, "spearman_rho": rho_A},
    {"comparison": "length vs prediction", "n": len(df_baseline), "pearson_r": r_B, "pearson_p": p_B, "spearman_rho": rho_B},
    {"comparison": "length vs abs_error", "n": len(df_baseline), "pearson_r": r_C, "pearson_p": p_C, "spearman_rho": rho_C},
    {"comparison": "length_change vs drift", "n": len(df_attacks), "pearson_r": r_D, "pearson_p": p_D, "spearman_rho": rho_D},
]
pd.DataFrame(stats_data).to_csv('results/final/shortcut_statistics.csv', index=False)

# Plots
plt.figure(figsize=(6,4))
plt.scatter(df_baseline['seq_len'], df_baseline['original_prediction'], alpha=0.5)
plt.title('Prediction vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Prediction (eV)')
plt.tight_layout()
plt.savefig('results/final/figures/prediction_vs_length.png')
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(df_baseline['seq_len'], df_baseline['absolute_error'], alpha=0.5)
plt.title('Error vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Absolute Error (eV)')
plt.tight_layout()
plt.savefig('results/final/figures/error_vs_length.png')
plt.close()

plt.figure(figsize=(6,4))
plt.scatter(df_attacks['length_change'], df_attacks['absolute_prediction_drift'], alpha=0.5)
plt.title('Drift vs Length Change')
plt.xlabel('Absolute Length Change')
plt.ylabel('Prediction Drift (eV)')
plt.tight_layout()
plt.savefig('results/final/figures/drift_vs_length_change.png')
plt.close()

with open('docs/SHORTCUT_ANALYSIS.md', 'w') as f:
    f.write("# SHORTCUT ANALYSIS\n\n")
    f.write("## Demonstrated Evidence\n")
    f.write(f"- Sequence length is weakly correlated with true target (Pearson r={r_A:.3f}).\n")
    f.write(f"- Sequence length is somewhat associated with the model's prediction (Pearson r={r_B:.3f}).\n")
    f.write(f"- Absolute length change has an association with absolute prediction drift (Pearson r={r_D:.3f}).\n")
    f.write("\n## Interpretation\n")
    f.write("Evidence is consistent with the model relying on sequence length as a partial proxy for properties, causing it to be sensitive to length-changing edits like insertions or deletions.\n")
