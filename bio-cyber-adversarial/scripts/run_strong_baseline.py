import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

df = pd.read_csv("data/v2/dataset.csv")
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

def extract_features(seqs):
    # 1-6 mer counts
    vec = CountVectorizer(analyzer='char', ngram_range=(1, 6), max_features=5000)
    X_counts = vec.fit_transform(seqs)
    
    # motif presence
    # ATGC and GCAT
    has_atgc = np.array([1 if "ATGC" in s else 0 for s in seqs]).reshape(-1, 1)
    has_gcat = np.array([1 if "GCAT" in s else 0 for s in seqs]).reshape(-1, 1)
    
    import scipy.sparse as sp
    X = sp.hstack([X_counts, has_atgc, has_gcat])
    return X, vec

X_train, vec = extract_features(train["sequence"])
y_train = train["label"].values

X_test, _ = extract_features(test["sequence"])
# Re-transform with fitted vec
X_test_counts = vec.transform(test["sequence"])
has_atgc_test = np.array([1 if "ATGC" in s else 0 for s in test["sequence"]]).reshape(-1, 1)
has_gcat_test = np.array([1 if "GCAT" in s else 0 for s in test["sequence"]]).reshape(-1, 1)
import scipy.sparse as sp
X_test_final = sp.hstack([X_test_counts, has_atgc_test, has_gcat_test])
y_test = test["label"].values

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test_final)
probs = clf.predict_proba(X_test_final)[:, 1]

acc = accuracy_score(y_test, preds)
bacc = balanced_accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

with open("results/v2_strong_baseline.txt", "w") as f:
    f.write(f"accuracy: {acc}\n")
    f.write(f"balanced accuracy: {bacc}\n")
    f.write(f"F1: {f1}\n")
    f.write(f"AUROC: {auc}\n")
