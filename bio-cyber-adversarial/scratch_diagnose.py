import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/dataset.csv')
print("--- Diagnosis ---")

# 1. Motif frequencies
motif_0 = "ACGTACGT"
motif_1 = "TGCATGCA"

df['has_m0'] = df['sequence'].str.contains(motif_0)
df['has_m1'] = df['sequence'].str.contains(motif_1)

print("\nMotif 0 counts by class:")
print(pd.crosstab(df['label'], df['has_m0']))

print("\nMotif 1 counts by class:")
print(pd.crosstab(df['label'], df['has_m1']))

# Let's check how many times Motif 1 occurs randomly in Class 0, and vice versa.
# Also, if Motif 0 happens to be easier to learn.
