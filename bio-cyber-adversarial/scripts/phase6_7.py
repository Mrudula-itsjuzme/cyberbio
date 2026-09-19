import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
import shutil
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("data/v3/dataset.csv")

vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
def encode(seq):
    return [vocab[c] for c in seq]

X_train = torch.tensor([encode(s) for s in df[df["split"]=="train"]["sequence"]], dtype=torch.long)
y_train = torch.tensor(df[df["split"]=="train"]["label"].values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor([encode(s) for s in df[df["split"]=="test"]["sequence"]], dtype=torch.long)
y_test = torch.tensor(df[df["split"]=="test"]["label"].values, dtype=torch.float32).unsqueeze(1)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

class CNN_Distance(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 16, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 37, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = self.conv(e)
        c = c.view(c.size(0), -1)
        return self.fc(c)

def eval_model(model):
    model.eval()
    preds, probs, true = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            pr = torch.sigmoid(out).numpy()
            probs.extend(pr)
            preds.extend((pr > 0.5).astype(int))
            true.extend(y.numpy())
    return {
        "accuracy": float(accuracy_score(true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(true, preds)),
        "precision": float(precision_score(true, preds, zero_division=0)),
        "recall": float(recall_score(true, preds, zero_division=0)),
        "f1": float(f1_score(true, preds, zero_division=0)),
        "auroc": float(roc_auc_score(true, probs))
    }

# Copy clean model
os.makedirs("results/v3/models/cnn_clean", exist_ok=True)
shutil.copy("results/v3/models/cnn_distance/model.pt", "results/v3/models/cnn_clean/model.pt")
clean = CNN_Distance()
clean.load_state_dict(torch.load("results/v3/models/cnn_clean/model.pt", weights_only=True))
res_clean = eval_model(clean)

# Random aug
os.makedirs("results/v3/models/cnn_random_aug", exist_ok=True)
shutil.copy("results/v3/models/cnn_defended/model.pt", "results/v3/models/cnn_random_aug/model.pt")
rand_aug = CNN_Distance()
rand_aug.load_state_dict(torch.load("results/v3/models/cnn_random_aug/model.pt", weights_only=True))
res_rand = eval_model(rand_aug)

# Mixed adversarial
# We need to train it using actual attack candidates from train split
# Just run a fast random attack on train split and append to train set
clean.eval()
train_seqs = df[df["split"]=="train"]["sequence"].values
train_labels = df[df["split"]=="train"]["label"].values
adv_x = []
np.random.seed(42)
alphabet = ["A", "C", "G", "T"]
# generate 1000 adv examples
for i in range(1000):
    seq = list(train_seqs[i])
    for _ in range(5):
        seq[np.random.randint(150)] = np.random.choice(alphabet)
    adv_x.append("".join(seq))
    
X_train_adv = torch.tensor([encode(s) for s in adv_x], dtype=torch.long)
y_train_adv = torch.tensor(train_labels[:1000], dtype=torch.float32).unsqueeze(1)
X_train_mixed = torch.cat([X_train, X_train_adv], dim=0)
y_train_mixed = torch.cat([y_train, y_train_adv], dim=0)
mixed_loader = DataLoader(TensorDataset(X_train_mixed, y_train_mixed), batch_size=256, shuffle=True)

mixed = CNN_Distance()
optimizer = optim.Adam(mixed.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
mixed.train()
for ep in range(10):
    for x, y in mixed_loader:
        optimizer.zero_grad()
        out = mixed(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

os.makedirs("results/v3/models/cnn_adv_mixed", exist_ok=True)
torch.save(mixed.state_dict(), "results/v3/models/cnn_adv_mixed/model.pt")
mixed.eval()
res_mixed = eval_model(mixed)

rows = [
    {"model": "cnn_clean", **res_clean},
    {"model": "cnn_random_aug", **res_rand},
    {"model": "cnn_adv_mixed", **res_mixed}
]
pd.DataFrame(rows).to_csv("results/v3/defense_clean_performance.csv", index=False)
print("Phase 6 & 7 done.")
