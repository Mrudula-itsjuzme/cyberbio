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

df = pd.read_csv('data/processed/processed.csv')
if 'split' in df.columns:
    test_df = df[df['split'] == 'test']
else:
    test_df = df.sample(frac=1, random_state=42)

smiles_list = test_df['original_representation'].dropna().tolist()
true_vals = test_df['property_value'].dropna().tolist() if 'property_value' in test_df.columns else [0]*len(smiles_list)

predictor = GraphMPNNPredictor(strict=False)

def predict_safe(smi):
    try:
        return predictor.predict(smi)
    except Exception as e:
        return np.nan

def generate_variants(smiles, n_variants=30):
    s_clean = smiles.replace('*', '[Lv]')
    mol = Chem.MolFromSmiles(s_clean)
    if mol is None: return [], []
    canon = Chem.MolToSmiles(mol, canonical=True)
    
    variants = []
    families = []
    
    for _ in range(30):
        new_order = list(range(mol.GetNumAtoms()))
        np.random.shuffle(new_order)
        renumbered = Chem.RenumberAtoms(mol, new_order)
        var_smi = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=False)
        m_var = Chem.MolFromSmiles(var_smi)
        if m_var and Chem.MolToSmiles(m_var, canonical=True) == canon:
            variants.append(var_smi.replace('[Lv]', '*'))
            families.append('random_atom_ordering')
            
    for i in range(1, min(20, mol.GetNumAtoms())):
        new_order = list(range(mol.GetNumAtoms()))
        new_order = new_order[i:] + new_order[:i]
        renumbered = Chem.RenumberAtoms(mol, new_order)
        var_smi = Chem.MolToSmiles(renumbered, canonical=False, isomericSmiles=False)
        m_var = Chem.MolFromSmiles(var_smi)
        if m_var and Chem.MolToSmiles(m_var, canonical=True) == canon:
            variants.append(var_smi.replace('[Lv]', '*'))
            families.append('shift')
            
    return variants, families

np.random.seed(42)

all_results = []
equiv_variants = []
cohort = []
true_y = []

# Collect 100 successful sources
for s, y in zip(smiles_list, true_vals):
    base_pred = predict_safe(s)
    if np.isnan(base_pred):
        continue
    
    variants, families = generate_variants(s)
    if not variants:
        continue
        
    cohort.append(s)
    true_y.append(y)
    
    base_error = abs(base_pred - y) if y != 0 else 0
    
    seen = {s}
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
        pred = predict_safe(var)
        if np.isnan(pred):
            continue
            
        drift = abs(pred - base_pred)
        drift_vals.append(drift)
        predictions.append(pred)
        
        equiv_variants.append({
            'source_smiles': s,
            'variant_smiles': var,
            'family': fam,
            'pred': pred,
            'drift': drift,
            'length_delta': len(var) - len(s)
        })
        
    if len(drift_vals) == 0:
        continue
        
    all_preds = [base_pred] + predictions
    centered_disagreement = np.max([abs(p - np.mean(all_preds)) for p in all_preds])
        
    max_drift = np.max(drift_vals)
    all_results.append({
        'smiles': s,
        'clean_error': base_error,
        'max_drift': max_drift,
        'centered_disagreement': centered_disagreement,
        'mean_drift': np.mean(drift_vals),
        'p90_drift': np.percentile(drift_vals, 90) if drift_vals else 0,
        'std_drift': np.std(drift_vals) if len(drift_vals) > 0 else 0
    })
    
    if len(all_results) >= 100:
        break

with open('results/representation_equivalence/cohort_manifest.json', 'w') as f:
    json.dump({"polymers": cohort}, f)

df_variants = pd.DataFrame(equiv_variants)
df_results = pd.DataFrame(all_results)

df_variants.to_csv('results/representation_equivalence/equivalent_variants.csv', index=False)
df_variants.to_csv('results/representation_equivalence/equivalence_validation.csv', index=False)
df_variants[['variant_smiles', 'pred']].to_csv('results/representation_equivalence/predictions.csv', index=False)
df_results.to_csv('results/representation_equivalence/source_robustness.csv', index=False)
df_results.to_csv('results/representation_equivalence/error_vs_invariance.csv', index=False)
if not df_variants.empty:
    df_variants.groupby('family')['drift'].mean().reset_index().to_csv('results/representation_equivalence/family_results.csv', index=False)
if not df_results.empty:
    df_results.sort_values('max_drift', ascending=False).head(10).to_csv('results/representation_equivalence/top_failures.csv', index=False)

mean_drift = df_variants['drift'].mean() if not df_variants.empty else 0
median_drift = df_variants['drift'].median() if not df_variants.empty else 0
p90_drift = df_variants['drift'].quantile(0.90) if not df_variants.empty else 0
p95_drift = df_variants['drift'].quantile(0.95) if not df_variants.empty else 0
max_drift = df_variants['drift'].max() if not df_variants.empty else 0

rho, p_val = spearmanr(df_results['clean_error'], df_results['max_drift']) if not df_results.empty and df_results['clean_error'].sum() > 0 else (0, 1)

print("--- METRICS ---")
print(f"Num Sources: {len(df_results)}")
print(f"Num Representations: {len(df_variants)}")
print(f"Mean Drift: {mean_drift:.4f}")
print(f"Median Drift: {median_drift:.4f}")
print(f"P90 Drift: {p90_drift:.4f}")
print(f"P95 Drift: {p95_drift:.4f}")
print(f"Max Drift: {max_drift:.4f}")
print(f"Correlation (rho): {rho:.4f}, p-value: {p_val:.4e}")

worst_case_failure_rate = (df_results['max_drift'] > 0.1).mean() if not df_results.empty else 0
print(f"Worst Case Failure Rate (>0.1): {worst_case_failure_rate:.4f}")

# Figures
try:
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
except: pass

