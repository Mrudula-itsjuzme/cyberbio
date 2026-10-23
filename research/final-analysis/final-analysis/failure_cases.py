import json
import pandas as pd

df_orig = pd.read_csv('materials-adversarial/data/processed/processed.csv')
rep_col = 'original_representation'
if rep_col not in df_orig.columns: rep_col = 'smiles'
target_dict = dict(zip(df_orig['polymer_id'], df_orig['property_value']))

# Load clean attack records
records = []
with open('results/materials/experiments/phase2_paired_benchmark/clean_attack_records.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r['validity_status'] == 'valid':
            records.append(r)

df_attacks = pd.DataFrame(records)
df_attacks['true_target'] = df_attacks['sample_id'].map(target_dict)
df_attacks = df_attacks.sort_values(by='absolute_prediction_drift', ascending=False)

strongest = df_attacks.iloc[0]
weakest = df_attacks.iloc[-1]

with open('docs/FAILURE_CASES.md', 'w') as f:
    f.write("# FAILURE CASE ANALYSIS\n\n")
    f.write("## Strongest Successful Adversarial Example (Baseline Model)\n")
    f.write(f"- **Sample ID:** `{strongest['sample_id']}`\n")
    f.write(f"- **Original Sequence:** `{strongest['original_representation']}`\n")
    f.write(f"- **Adversarial Sequence:** `{strongest['adversarial_representation']}`\n")
    f.write(f"- **Validity Status:** `{strongest['validity_status']}`\n")
    f.write(f"- **True Target:** {strongest['true_target']:.4f} eV\n")
    f.write(f"- **Original Prediction:** {strongest['original_prediction']:.4f} eV\n")
    f.write(f"- **Adversarial Prediction:** {strongest['adversarial_prediction']:.4f} eV\n")
    f.write(f"- **Absolute Drift:** {strongest['absolute_prediction_drift']:.4f} eV\n")
    f.write(f"- **Attack Family:** {strongest['attack_type'].capitalize()}\n")
    f.write(f"- **Source File:** `results/materials/experiments/phase2_paired_benchmark/clean_attack_records.jsonl`\n")

    f.write("\n## Weakest Adversarial Example / Failure to Attack\n")
    f.write(f"- **Sample ID:** `{weakest['sample_id']}`\n")
    f.write(f"- **Original Sequence:** `{weakest['original_representation']}`\n")
    f.write(f"- **Adversarial Sequence:** `{weakest['adversarial_representation']}`\n")
    f.write(f"- **Validity Status:** `{weakest['validity_status']}`\n")
    f.write(f"- **True Target:** {weakest['true_target']:.4f} eV\n")
    f.write(f"- **Original Prediction:** {weakest['original_prediction']:.4f} eV\n")
    f.write(f"- **Adversarial Prediction:** {weakest['adversarial_prediction']:.4f} eV\n")
    f.write(f"- **Absolute Drift:** {weakest['absolute_prediction_drift']:.4f} eV\n")
    f.write(f"- **Attack Family:** {weakest['attack_type'].capitalize()}\n")
    f.write(f"- **Source File:** `results/materials/experiments/phase2_paired_benchmark/clean_attack_records.jsonl`\n")
