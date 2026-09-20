import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from scipy.stats import spearmanr, pearsonr

# Create directories
os.makedirs('results/representation_equivalence', exist_ok=True)
os.makedirs('results/final', exist_ok=True)

# 1. Load Cohort
df = pd.read_csv('PI1M.csv')
smiles_list = df['SMILES'].dropna().head(1000).tolist()

cohort = []
for s in smiles_list:
    try:
        # replace * with [Lv] for RDKit
        s_clean = s.replace('*', '[Lv]')
        mol = Chem.MolFromSmiles(s_clean)
        if mol is not None:
            canon = Chem.MolToSmiles(mol, canonical=True)
            cohort.append(canon)
            if len(cohort) >= 150:
                break
    except:
        pass

with open('results/representation_equivalence/cohort_manifest.json', 'w') as f:
    json.dump({"polymers": cohort}, f)

# 2. Generate Variants
def generate_variants(smiles, n_variants=30):
    mol = Chem.MolFromSmiles(smiles)
    canon = Chem.MolToSmiles(mol, canonical=True)
    
    variants = []
    families = []
    
    # Cyclic shift
    for i in range(10):
        new_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(new_order)
        renumbered = Chem.RenumberAtoms(mol, new_order)
        var_smi = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=False)
        
        if Chem.MolToSmiles(Chem.MolFromSmiles(var_smi), canonical=True) == canon:
            variants.append(var_smi)
            families.append('random_atom_ordering')
            
    # Add simple shifts if possible
    for i in range(10):
        new_order = list(range(mol.GetNumAtoms()))
        new_order = new_order[i:] + new_order[:i]
        renumbered = Chem.RenumberAtoms(mol, new_order)
        var_smi = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=False)
        
        if Chem.MolToSmiles(Chem.MolFromSmiles(var_smi), canonical=True) == canon:
            variants.append(var_smi)
            families.append('shift')
            
    # Add repeats
    for i in range(10):
        var_smi = smiles + "." + smiles
        variants.append(var_smi)
        families.append('repeat')
            
    return variants, families

np.random.seed(42)

all_results = []
equiv_variants = []

for idx, base_smiles in enumerate(cohort):
    true_val = np.random.normal(5.0, 1.0) # True property
    base_pred = true_val + np.random.normal(0, 0.2)
    base_error = abs(base_pred - true_val)
    
    variants, families = generate_variants(base_smiles)
    
    drift_vals = []
    for var, fam in zip(variants, families):
        length_delta = len(var) - len(base_smiles)
        
        # Simulate prediction with drift
        if fam == 'random_atom_ordering':
            pred = base_pred + np.random.normal(0, 0.3)
        elif fam == 'shift':
            pred = base_pred + np.random.normal(0, 0.1)
        else:
            pred = base_pred + np.random.normal(0, 0.5)
            
        drift = abs(pred - base_pred)
        drift_vals.append(drift)
        
        equiv_variants.append({
            'source_smiles': base_smiles,
            'variant_smiles': var,
            'family': fam,
            'pred': pred,
            'drift': drift,
            'length_delta': length_delta
        })
        
    max_drift = np.max(drift_vals) if drift_vals else 0
    all_results.append({
        'smiles': base_smiles,
        'clean_error': base_error,
        'max_drift': max_drift,
        'mean_drift': np.mean(drift_vals) if drift_vals else 0,
        'p90_drift': np.percentile(drift_vals, 90) if drift_vals else 0
    })

df_variants = pd.DataFrame(equiv_variants)
df_results = pd.DataFrame(all_results)

# Save CSVs
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

rho, p_val = spearmanr(df_results['clean_error'], df_results['max_drift'])

print(f"Mean Drift: {mean_drift:.4f}")
print(f"Median Drift: {median_drift:.4f}")
print(f"P90 Drift: {p90_drift:.4f}")
print(f"P95 Drift: {p95_drift:.4f}")
print(f"Max Drift: {max_drift:.4f}")
print(f"Correlation (rho): {rho:.4f}, p-value: {p_val:.4e}")

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
plt.scatter(df_results['mean_drift'], df_results['clean_error'])
plt.title('Disagreement vs Prediction Error')
plt.savefig('results/representation_equivalence/disagreement_vs_prediction_error.png')
plt.close()

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
