import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import os

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

df = pd.read_csv("comparison-experiments/results/summaries/budget_sweep_results.csv")

def get_tanimoto(s1, s2):
    try:
        m1 = Chem.MolFromSmiles(s1)
        m2 = Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None: return 0.0
        fp1 = FingerprintMols.FingerprintMol(m1)
        fp2 = FingerprintMols.FingerprintMol(m2)
        return DataStructs.FingerprintSimilarity(fp1, fp2)
    except:
        return 0.0

def get_validity(s):
    try:
        return Chem.MolFromSmiles(s) is not None
    except:
        return False

print("Recomputing metrics for 4500 candidates...")
df['recomputed_valid_rdkit'] = df['candidate_sequence'].apply(get_validity)
df['recomputed_edit_distance'] = df.apply(lambda row: edit_distance(str(row['source_sequence']), str(row['candidate_sequence'])), axis=1)
df['recomputed_tanimoto'] = df.apply(lambda row: get_tanimoto(str(row['source_sequence']), str(row['candidate_sequence'])), axis=1)

os.makedirs("comparison-experiments/results/recomputed_metrics", exist_ok=True)
df.to_csv("comparison-experiments/results/recomputed_metrics/classical_candidate_metrics.csv", index=False)
print("Finished recomputing classical metrics.")
