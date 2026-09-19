import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/v2/dataset.csv")

train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

y_train = train_df["label"].values
y_test = test_df["label"].values
texts_train = train_df["sequence"].values
texts_test = test_df["sequence"].values

results = {}

def evaluate_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0))
    }

def char_ngrams(texts, n):
    vec = CountVectorizer(analyzer='char', ngram_range=(n, n))
    X = vec.fit_transform(texts)
    return X, vec

# 1-mer to 5-mer
for n in [1, 2, 3, 4, 5]:
    X_tr_n, vec = char_ngrams(texts_train, n)
    X_te_n = vec.transform(texts_test)
    
    lr = LogisticRegression(max_iter=1000)
    res_lr = evaluate_model(lr, X_tr_n, y_train, X_te_n, y_test)
    results[f"{n}-mer_LogisticRegression"] = res_lr

# Position-aware baseline
def extract_pos_features(texts):
    feats = []
    for t in texts:
        p1 = t.find("ATGC")
        p2 = t.find("GCAT")
        if p1 != -1 and p2 != -1:
            feats.append([abs(p2 - p1)])
        else:
            feats.append([0])
    return np.array(feats)

X_tr_pos = extract_pos_features(texts_train)
X_te_pos = extract_pos_features(texts_test)

lr_pos = LogisticRegression()
results["position_aware_baseline"] = evaluate_model(lr_pos, X_tr_pos, y_train, X_te_pos, y_test)

with open("results/v2_leakage_audit_extended.json", "w") as f:
    json.dump(results, f, indent=4)

print("V2 extended shortcut check complete.")
