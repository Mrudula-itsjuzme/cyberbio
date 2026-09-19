import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
import json

df = pd.read_csv("data/v3/dataset.csv")
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

results = {}

for k in range(1, 7):
    vec = CountVectorizer(analyzer='char', ngram_range=(k, k), max_features=5000)
    X_train = vec.fit_transform(train["sequence"])
    X_test = vec.transform(test["sequence"])
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train["label"])
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    results[f"{k}-mer"] = {
        "accuracy": accuracy_score(test["label"], preds),
        "f1": f1_score(test["label"], preds),
        "auroc": roc_auc_score(test["label"], probs)
    }

# Strong shallow baseline
def strong_features(seqs):
    vec = CountVectorizer(analyzer='char', ngram_range=(1, 6), max_features=5000)
    X_counts = vec.fit_transform(seqs)
    has_atgc = np.array([1 if "ATGC" in s else 0 for s in seqs]).reshape(-1, 1)
    has_gcat = np.array([1 if "GCAT" in s else 0 for s in seqs]).reshape(-1, 1)
    import scipy.sparse as sp
    return sp.hstack([X_counts, has_atgc, has_gcat]), vec

X_train_s, vec_s = strong_features(train["sequence"])
X_test_s = vec_s.transform(test["sequence"])
has_atgc_test = np.array([1 if "ATGC" in s else 0 for s in test["sequence"]]).reshape(-1, 1)
has_gcat_test = np.array([1 if "GCAT" in s else 0 for s in test["sequence"]]).reshape(-1, 1)
import scipy.sparse as sp
X_test_final = sp.hstack([X_test_s, has_atgc_test, has_gcat_test])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, train["label"])
preds = clf.predict(X_test_final)
probs = clf.predict_proba(X_test_final)[:, 1]

results["strong_shallow"] = {
    "accuracy": accuracy_score(test["label"], preds),
    "f1": f1_score(test["label"], preds),
    "auroc": roc_auc_score(test["label"], probs)
}

with open("results/v3_shortcut_gate.json", "w") as f:
    json.dump(results, f, indent=4)
print("V3 Shortcut gate executed.")
