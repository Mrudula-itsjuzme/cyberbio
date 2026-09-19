import pandas as pd
import numpy as np
import json
import os
import re
from collections import Counter

df = pd.read_csv("data/raw/dataset.csv")

# Train/test split based on 'split' column
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

def tokenize(s): return list(s)
def ngrams(tokens, n): return ["".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def build_features(texts, n=1, vocab=None):
    list_of_tokens = [tokenize(t) for t in texts]
    if n > 1:
        list_of_tokens = [ngrams(toks, n) for toks in list_of_tokens]
    if vocab is None:
        vocab = set()
        for toks in list_of_tokens: vocab.update(toks)
        vocab = sorted(list(vocab))
    v2i = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for i, toks in enumerate(list_of_tokens):
        counts = Counter(toks)
        for k, v in counts.items():
            if k in v2i: X[i, v2i[k]] = v
    return X, vocab

def logistic_regression(X_train, y_train, X_test, y_test, alpha=1.0):
    # simple ridge classifier via least squares
    X_tr = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_te = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    I = np.eye(X_tr.shape[1]); I[-1, -1] = 0
    w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_train)
    
    preds_tr = (X_tr @ w) > 0.5
    preds_te = (X_te @ w) > 0.5
    acc_tr = np.mean(preds_tr == y_train)
    acc_te = np.mean(preds_te == y_test)
    return float(acc_tr), float(acc_te)

y_tr = train_df["label"].values
y_te = test_df["label"].values

print("Running bio-cyber leakage audit...")
res = {}

# 1. Sequence length
X_tr_len = train_df["sequence"].apply(len).values.reshape(-1, 1)
X_te_len = test_df["sequence"].apply(len).values.reshape(-1, 1)
res["sequence_length"] = logistic_regression(X_tr_len, y_tr, X_te_len, y_te)

# 2. 1-mer
X_tr_1, v1 = build_features(train_df["sequence"].values, n=1)
X_te_1, _ = build_features(test_df["sequence"].values, n=1, vocab=v1)
res["1-mer"] = logistic_regression(X_tr_1, y_tr, X_te_1, y_te)

# 3. 2-mer
X_tr_2, v2 = build_features(train_df["sequence"].values, n=2)
X_te_2, _ = build_features(test_df["sequence"].values, n=2, vocab=v2)
res["2-mer"] = logistic_regression(X_tr_2, y_tr, X_te_2, y_te)

# 4. 3-mer
X_tr_3, v3 = build_features(train_df["sequence"].values, n=3)
X_te_3, _ = build_features(test_df["sequence"].values, n=3, vocab=v3)
res["3-mer"] = logistic_regression(X_tr_3, y_tr, X_te_3, y_te)

os.makedirs("results", exist_ok=True)
with open("results/leakage_audit.json", "w") as f:
    json.dump(res, f, indent=4)
print(res)
print("Phase H Complete.")
