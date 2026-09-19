import pandas as pd
import numpy as np
import os
import json
import re
from collections import Counter

df = pd.read_csv("materials-adversarial/data/processed/processed.csv")
with open("materials-adversarial/data/processed/splits.json", "r") as f:
    splits = json.load(f)

# The tokenizer from the codebase is available via PYTHONPATH, but since we are running isolated:
def tokenize(s):
    # Regex matching bracketed tokens or single chars
    return re.findall(r'\[[^\]]+\]|.', s)

def ngrams(tokens, n):
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def build_features(texts, n=1, vocab=None, normalize=False):
    list_of_tokens = [tokenize(t) for t in texts]
    if n > 1:
        list_of_tokens = [ngrams(toks, n) for toks in list_of_tokens]
    
    if vocab is None:
        vocab = set()
        for toks in list_of_tokens:
            vocab.update(toks)
        vocab = sorted(list(vocab))
        
    v2i = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for i, toks in enumerate(list_of_tokens):
        counts = Counter(toks)
        for k, v in counts.items():
            if k in v2i:
                X[i, v2i[k]] = v
                
    if normalize:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        X = X / row_sums
        
    return X, vocab

def ridge_regression(X_train, y_train, X_val, y_val, X_test, y_test, alpha=1.0):
    # Add intercept
    X_tr = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_va = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_te = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    
    # Ridge: w = (X^T X + alpha I)^-1 X^T y
    I = np.eye(X_tr.shape[1])
    I[-1, -1] = 0 # don't regularize intercept
    w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_train)
    
    def r2_rmse_mae(X, y):
        preds = X @ w
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((y - preds)**2))
        mae = np.mean(np.abs(y - preds))
        return r2, rmse, mae
        
    r2_tr, rmse_tr, mae_tr = r2_rmse_mae(X_tr, y_train)
    r2_va, rmse_va, mae_va = r2_rmse_mae(X_va, y_val)
    r2_te, rmse_te, mae_te = r2_rmse_mae(X_te, y_test)
    return {
        "train": {"r2": r2_tr, "rmse": rmse_tr, "mae": mae_tr},
        "val": {"r2": r2_va, "rmse": rmse_va, "mae": mae_va},
        "test": {"r2": r2_te, "rmse": rmse_te, "mae": mae_te}
    }

train_df = df.iloc[splits["train"]]
val_df = df.iloc[splits["val"]]
test_df = df.iloc[splits["test"]]

y_train = train_df["property_value"].values
y_val = val_df["property_value"].values
y_test = test_df["property_value"].values

results = {}

print("1. Sequence Length")
X_tr_len = train_df["original_representation"].apply(len).values.reshape(-1, 1)
X_va_len = val_df["original_representation"].apply(len).values.reshape(-1, 1)
X_te_len = test_df["original_representation"].apply(len).values.reshape(-1, 1)
results["sequence_length"] = ridge_regression(X_tr_len, y_train, X_va_len, y_val, X_te_len, y_test, alpha=0.0)

print("2. Token Counts (Bag of Tokens)")
X_tr_bow, vocab_1 = build_features(train_df["original_representation"].values, n=1)
X_va_bow, _ = build_features(val_df["original_representation"].values, n=1, vocab=vocab_1)
X_te_bow, _ = build_features(test_df["original_representation"].values, n=1, vocab=vocab_1)
results["token_counts"] = ridge_regression(X_tr_bow, y_train, X_va_bow, y_val, X_te_bow, y_test, alpha=1.0)

print("3. Normalized Token Frequencies")
X_tr_freq, vocab_1_f = build_features(train_df["original_representation"].values, n=1, normalize=True)
X_va_freq, _ = build_features(val_df["original_representation"].values, n=1, vocab=vocab_1_f, normalize=True)
X_te_freq, _ = build_features(test_df["original_representation"].values, n=1, vocab=vocab_1_f, normalize=True)
results["token_frequencies"] = ridge_regression(X_tr_freq, y_train, X_va_freq, y_val, X_te_freq, y_test, alpha=1.0)

print("4. 2-gram Counts")
X_tr_2g, vocab_2 = build_features(train_df["original_representation"].values, n=2)
X_va_2g, _ = build_features(val_df["original_representation"].values, n=2, vocab=vocab_2)
X_te_2g, _ = build_features(test_df["original_representation"].values, n=2, vocab=vocab_2)
results["2-gram_counts"] = ridge_regression(X_tr_2g, y_train, X_va_2g, y_val, X_te_2g, y_test, alpha=10.0)

print("5. 3-gram Counts")
X_tr_3g, vocab_3 = build_features(train_df["original_representation"].values, n=3)
X_va_3g, _ = build_features(val_df["original_representation"].values, n=3, vocab=vocab_3)
X_te_3g, _ = build_features(test_df["original_representation"].values, n=3, vocab=vocab_3)
results["3-gram_counts"] = ridge_regression(X_tr_3g, y_train, X_va_3g, y_val, X_te_3g, y_test, alpha=50.0)

# Correlations
def calc_corr(feature_func):
    feats = train_df["original_representation"].apply(feature_func).values
    # Pearson R
    if np.std(feats) == 0: return 0.0
    return np.corrcoef(feats, y_train)[0, 1]

correlations = {
    "sequence_length": calc_corr(len),
    "attachment_point_count": calc_corr(lambda s: s.count('*')),
    "ring_character_count": calc_corr(lambda s: sum(c.isdigit() for c in s)),
    "heteroatom_count": calc_corr(lambda s: sum(c in 'NOSPF' for c in s.upper()))
}

out_dir = "comparison-experiments/results/shortcut_audit"
os.makedirs(out_dir, exist_ok=True)

with open(f"{out_dir}/shortcut_baselines.json", "w") as f:
    json.dump({"baselines": results, "correlations": correlations}, f, indent=4)

df_out = []
for k, v in results.items():
    df_out.append({
        "baseline": k,
        "train_R2": v["train"]["r2"],
        "val_R2": v["val"]["r2"],
        "test_R2": v["test"]["r2"],
        "test_RMSE": v["test"]["rmse"]
    })
pd.DataFrame(df_out).to_csv(f"{out_dir}/shortcut_baselines.csv", index=False)
print("Phase A Complete.")
