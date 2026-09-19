import pandas as pd
import numpy as np
import json
import os
from collections import Counter
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("data/raw/dataset.csv")

train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()
test_df = df[df['split'] == 'test'].copy()

# Integrity Audit
print("Running Integrity Audit...")
split_integrity = {}

# Exact duplicate sequences
all_seqs = df['sequence'].tolist()
counts = Counter(all_seqs)
duplicates = sum(1 for v in counts.values() if v > 1)

# Duplicates across splits
train_seqs = set(train_df['sequence'])
test_seqs = set(test_df['sequence'])
val_seqs = set(val_df['sequence'])
overlap_train_test = len(train_seqs.intersection(test_seqs))
overlap_train_val = len(train_seqs.intersection(val_seqs))

# Class-dependent length
lengths_0 = train_df[train_df['label'] == 0]['sequence'].apply(len)
lengths_1 = train_df[train_df['label'] == 1]['sequence'].apply(len)

# Motif bias
motif_pos_0 = train_df[train_df['label'] == 0]['motif_pos'].mean()
motif_pos_1 = train_df[train_df['label'] == 1]['motif_pos'].mean()

split_integrity["exact_duplicate_sequences_total"] = duplicates
split_integrity["cross_split_duplicates_train_test"] = overlap_train_test
split_integrity["cross_split_duplicates_train_val"] = overlap_train_val
split_integrity["mean_length_class_0"] = float(lengths_0.mean())
split_integrity["mean_length_class_1"] = float(lengths_1.mean())
split_integrity["mean_motif_pos_class_0"] = float(motif_pos_0)
split_integrity["mean_motif_pos_class_1"] = float(motif_pos_1)

# Categorize issues
findings = []
if overlap_train_test > 0 or overlap_train_val > 0:
    findings.append({"issue": "Duplicates across splits", "category": "SPLIT_LEAKAGE"})
if abs(lengths_0.mean() - lengths_1.mean()) > 1.0:
    findings.append({"issue": "Class-dependent sequence length", "category": "SHORTCUT_FEATURE"})
if duplicates > 0:
    findings.append({"issue": "High exact duplicates", "category": "GENERATOR_ARTIFACT"})
if len(findings) == 0:
    findings.append({"issue": "None", "category": "NO_ISSUE_FOUND"})

split_integrity["findings"] = findings
os.makedirs("results", exist_ok=True)
with open("results/split_integrity_audit.json", "w") as f:
    json.dump(split_integrity, f, indent=4)


# Leakage Audit via Sklearn
print("Running Leakage Audit (LR & Ridge)...")

def char_ngrams(texts, n):
    vec = CountVectorizer(analyzer='char', ngram_range=(n, n))
    X = vec.fit_transform(texts)
    return X, vec

def evaluate_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

y_train = train_df["label"].values
y_test = test_df["label"].values
texts_train = train_df["sequence"].values
texts_test = test_df["sequence"].values

results = {}

# Baseline 1: Majority Class
maj_class = int(train_df["label"].mode()[0])
y_pred_maj = [maj_class] * len(y_test)
results["majority_class"] = {
    "accuracy": float(accuracy_score(y_test, y_pred_maj)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_maj)),
    "f1": float(f1_score(y_test, y_pred_maj, zero_division=0)),
}

# Baseline 2: Sequence Length
X_tr_len = np.array([len(t) for t in texts_train]).reshape(-1, 1)
X_te_len = np.array([len(t) for t in texts_test]).reshape(-1, 1)
lr_len = LogisticRegression()
results["sequence_length_LR"] = evaluate_model(lr_len, X_tr_len, y_train, X_te_len, y_test)

# K-mers
for n in [1, 2, 3, 4]:
    X_tr_n, vec = char_ngrams(texts_train, n)
    X_te_n = vec.transform(texts_test)
    
    lr = LogisticRegression(max_iter=1000)
    res_lr = evaluate_model(lr, X_tr_n, y_train, X_te_n, y_test)
    results[f"{n}-mer_LogisticRegression"] = res_lr
    
    ridge = RidgeClassifier()
    res_ridge = evaluate_model(ridge, X_tr_n.toarray() if n <= 2 else X_tr_n, y_train, X_te_n.toarray() if n <= 2 else X_te_n, y_test)
    results[f"{n}-mer_RidgeClassifier"] = res_ridge

with open("results/leakage_audit.json", "w") as f:
    json.dump(results, f, indent=4)
print("Steps 1 and 2 complete.")
