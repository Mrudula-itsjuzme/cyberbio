import pandas as pd
import numpy as np
import os
import json
from scipy.stats import pearsonr

df = pd.read_csv("materials-adversarial/data/processed/processed.csv")
with open("materials-adversarial/data/processed/splits.json", "r") as f:
    splits = json.load(f)

train_df = df.iloc[splits["train"]]
val_df = df.iloc[splits["val"]]

# Baseline 1: Sequence Length
train_len = train_df["original_representation"].apply(len).values
val_len = val_df["original_representation"].apply(len).values

y_train = train_df["property_value"].values
y_val = val_df["property_value"].values

# Simple linear regression using numpy
A = np.vstack([train_len, np.ones(len(train_len))]).T
m, c = np.linalg.lstsq(A, y_train, rcond=None)[0]

preds_len = m * val_len + c
# Calculate R2
ss_res = np.sum((y_val - preds_len) ** 2)
ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
r2_len = 1 - (ss_res / ss_tot)

print(f"Sequence Length R2: {r2_len:.4f}")

# Baseline 2: Token Count / Bag of Tokens
from collections import Counter
def tokenize(s): return list(s)

def build_bow(texts, vocab=None):
    if vocab is None:
        vocab = set()
        for t in texts:
            vocab.update(tokenize(t))
        vocab = sorted(list(vocab))
    v2i = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        counts = Counter(tokenize(t))
        for k, v in counts.items():
            if k in v2i:
                X[i, v2i[k]] = v
    return X, vocab

X_train_bow, vocab = build_bow(train_df["original_representation"].values)
X_val_bow, _ = build_bow(val_df["original_representation"].values, vocab)

# Multi-variate regression
A_bow = np.hstack([X_train_bow, np.ones((X_train_bow.shape[0], 1))])
w = np.linalg.lstsq(A_bow, y_train, rcond=None)[0]

A_val_bow = np.hstack([X_val_bow, np.ones((X_val_bow.shape[0], 1))])
preds_bow = A_val_bow @ w

ss_res_bow = np.sum((y_val - preds_bow) ** 2)
r2_bow = 1 - (ss_res_bow / ss_tot)

os.makedirs("comparison-experiments/results/shortcut_audit", exist_ok=True)
with open("comparison-experiments/results/shortcut_audit/baseline_r2.json", "w") as f:
    json.dump({
        "sequence_length_r2": float(r2_len),
        "bag_of_tokens_r2": float(r2_bow),
        "vocab_size": len(vocab)
    }, f, indent=4)

print(f"Bag-of-Tokens R2: {r2_bow:.4f}")
