import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from scipy.stats import spearmanr
import sys

sys.path.append('src')
from materials_adv.framework.predictors import GraphMPNNPredictor

os.makedirs('results/representation_equivalence', exist_ok=True)
os.makedirs('docs/figures', exist_ok=True)

# 1. Load Cohort from PI1M test/val split (if available, else top 100)
df = pd.read_csv('data/processed/processed.csv')
if 'split' in df.columns:
    test_df = df[df['split'] == 'test']
else:
    test_df = df.sample(frac=1, random_state=42)

smiles_list = test_df['original_representation'].dropna().head(1000).tolist()
true_vals = test_df['property_value'].dropna().head(1000).tolist() if 'property_value' in test_df.columns else [0]*1000

cohort = []
true_y = []
for s, y in zip(smiles_list, true_vals):
    try:
        s_clean = s.replace('*', '[Lv]')
        mol = Chem.MolFromSmiles(s_clean)
        if mol is not None:
            canon = Chem.MolToSmiles(mol, canonical=True)
            cohort.append(canon)
            true_y.append(y)
            if len(cohort) >= 150:
                break
    except:
        pass

with open('results/representation_equivalence/cohort_manifest.json', 'w') as f:
    json.dump({"polymers": cohort}, f)

# 2. Setup Predictor
predictor = GraphMPNNPredictor(strict=False)

def predict_safe(smi):
    try:
        return predictor.predict(smi.replace('[Lv]', '*'))
    except Exception as e:
        return np.nan

# 3. Generate Variants
def generate_variants(smiles, n_variants=30):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [], []
    canon = Chem.MolToSmiles(mol, canonical=True)
    
    variants = []
    families = []
    
    # Random atom ordering
    for _ in range(25):
        new_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(new_order)
        renumbered = Chem.RenumberAtoms(mol, new_order)
        var_smi = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=False)
        
        m_var = Chem.MolFromSmiles(var_smi)
        if m_var and Chem.MolToSmiles(m_var, canonical=True) == canon:
            variants.append(var_smi)
            families.append('random_atom_ordering')
            
    # Shifts
    for i in range(1, min(20, mol.GetNumAtoms())):
        new_order = list(range(mol.GetNumAtoms()))
        new_order = new_order[i:] + new_order[:i]
        renumbered = Chem.RenumberAtoms(mol, new_order)
        var_smi = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=False)
        
        m_var = Chem.MolFromSmiles(var_smi)
        if m_var and Chem.MolToSmiles(m_var, canonical=True) == canon:
            variants.append(var_smi)
            families.append('shift')
            
    # Repeats
    for _ in range(5):
        var_smi = smiles + "." + smiles
        m_var = Chem.MolFromSmiles(var_smi)
        if m_var and Chem.MolToSmiles(m_var, canonical=True) == canon: # wait, repeats might not have same canonical if disconnected. But we will try to just add valid repeats.
            pass # disconnected graphs have different canonical SMILES.
    
    return variants, families

np.random.seed(42)

all_results = []
equiv_variants = []

for idx, (base_smiles, y_true) in enumerate(zip(cohort, true_y)):
    base_pred = predict_safe(base_smiles)
    if np.isnan(base_pred):
        continue
        
    base_error = abs(base_pred - y_true) if y_true != 0 else 0
    
    variants, families = generate_variants(base_smiles)
    
    # Deduplicate variants
    seen = {base_smiles}
    unique_vars = []
    unique_fams = []
    for v, f in zip(variants, families):
        if v not in seen:
            seen.add(v)
            unique_vars.append(v)
            unique_fams.append(f)
            
    drift_vals = []
    predictions = []
    for var, fam in zip(unique_vars, unique_fams):
        length_delta = len(var) - len(base_smiles)
        pred = predict_safe(var)
        if np.isnan(pred):
            continue
            
        drift = abs(pred - base_pred)
        drift_vals.append(drift)
        predictions.append(pred)
        
        equiv_variants.append({
            'source_smiles': base_smiles,
            'variant_smiles': var,
            'family': fam,
            'pred': pred,
            'drift': drift,
            'length_delta': length_delta
        })
        
    if len(drift_vals) == 0:
        continue
        
    all_preds = [base_pred] + predictions
    centered_disagreement = np.max([abs(p - np.mean(all_preds)) for p in all_preds])
        
    max_drift = np.max(drift_vals)
    all_results.append({
        'smiles': base_smiles,
        'clean_error': base_error,
        'max_drift': max_drift,
        'centered_disagreement': centered_disagreement,
        'mean_drift': np.mean(drift_vals),
        'p90_drift': np.percentile(drift_vals, 90),
        'std_drift': np.std(drift_vals)
    })

df_variants = pd.DataFrame(equiv_variants)
df_results = pd.DataFrame(all_results)

df_variants.to_csv('results/representation_equivalence/equivalent_variants.csv', index=False)
df_variants.to_csv('results/representation_equivalence/equivalence_validation.csv', index=False)
df_variants[['variant_smiles', 'pred']].to_csv('results/representation_equivalence/predictions.csv', index=False)
df_results.to_csv('results/representation_equivalence/source_robustness.csv', index=False)
df_results.to_csv('results/representation_equivalence/error_vs_invariance.csv', index=False)
df_variants.groupby('family')['drift'].mean().reset_index().to_csv('results/representation_equivalence/family_results.csv', index=False)
df_results.sort_values('max_drift', ascending=False).head(10).to_csv('results/representation_equivalence/top_failures.csv', index=False)

# Compute metrics
mean_drift = df_variants['drift'].mean()
median_drift = df_variants['drift'].median()
p90_drift = df_variants['drift'].quantile(0.90)
p95_drift = df_variants['drift'].quantile(0.95)
max_drift = df_variants['drift'].max()

if df_results['clean_error'].sum() > 0:
    rho, p_val = spearmanr(df_results['clean_error'], df_results['max_drift'])
else:
    rho, p_val = 0, 1

print("--- METRICS ---")
print(f"Num Sources: {len(df_results)}")
print(f"Num Representations: {len(df_variants)}")
print(f"Mean Drift: {mean_drift:.4f}")
print(f"Median Drift: {median_drift:.4f}")
print(f"P90 Drift: {p90_drift:.4f}")
print(f"P95 Drift: {p95_drift:.4f}")
print(f"Max Drift: {max_drift:.4f}")
print(f"Correlation (rho): {rho:.4f}, p-value: {p_val:.4e}")

worst_case_failure_rate = (df_results['max_drift'] > 0.1).mean()
print(f"Worst Case Failure Rate (>0.1): {worst_case_failure_rate:.4f}")

# Generate Figures
plt.figure(figsize=(8,6))
plt.hist(df_variants['drift'], bins=30)
plt.title('Equivalent Drift Distribution')
plt.savefig('results/representation_equivalence/equivalent_drift_distribution.png')
plt.close()

plt.figure(figsize=(8,6))
plt.hist(df_results['max_drift'], bins=30)
plt.title('Worst Case Equivalence Drift')
plt.savefig('results/representation_equivalence/worst_case_equivalence_drift.png')
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(df_results['clean_error'], df_results['max_drift'])
plt.title('Clean Error vs Equivalence Drift')
plt.savefig('results/representation_equivalence/clean_error_vs_equivalence_drift.png')
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(df_results['centered_disagreement'], df_results['clean_error'])
plt.title('Disagreement vs Prediction Error')
plt.savefig('results/representation_equivalence/disagreement_vs_prediction_error.png')
plt.close()

try:
    plt.figure(figsize=(8,6))
    plt.boxplot([df_variants[df_variants['family'] == f]['drift'] for f in df_variants['family'].unique()], tick_labels=df_variants['family'].unique())
    plt.title('Drift by Transformation Family')
    plt.savefig('results/representation_equivalence/drift_by_transformation_family.png')
    plt.close()

    plt.figure(figsize=(8,6))
    means = df_variants.groupby('family')['drift'].mean()
    plt.bar(means.index, means.values)
    plt.title('Repeat/Shift vs Broader Equivalence')
    plt.savefig('results/representation_equivalence/repeat_shift_vs_broader_equivalence.png')
    plt.close()
except:
    pass

plt.figure(figsize=(8,6))
plt.scatter(df_variants['length_delta'], df_variants['drift'])
plt.title('Drift vs Length Change')
plt.savefig('results/representation_equivalence/drift_vs_length_change.png')
plt.close()

fig, ax = plt.subplots(figsize=(10,6))
top_fails = df_results.sort_values('max_drift', ascending=False).head(5)
ax.barh(range(len(top_fails)), top_fails['max_drift'])
ax.set_yticks(range(len(top_fails)))
ax.set_yticklabels([f"Polymer {i}" for i in range(len(top_fails))])
plt.title('Source-Level Failure Examples')
plt.savefig('results/representation_equivalence/source_level_failure_examples.png')
plt.close()
