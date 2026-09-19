import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/v3/dataset.csv")
vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
def encode(seq): return [vocab[c] for c in seq]

X_train = torch.tensor([encode(s) for s in df[df["split"]=="train"]["sequence"]], dtype=torch.long)
y_train = torch.tensor(df[df["split"]=="train"]["label"].values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor([encode(s) for s in df[df["split"]=="test"]["sequence"]], dtype=torch.long)
y_test = torch.tensor(df[df["split"]=="test"]["label"].values, dtype=torch.float32).unsqueeze(1)

class CNN_Distance(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 16, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(nn.Linear(64 * 37, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = self.conv(e)
        return self.fc(c.view(c.size(0), -1))

def train_eval(seed, shuffle=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN_Distance()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    
    y_tr = y_train
    if shuffle:
        y_tr = y_train[torch.randperm(y_train.size(0))]
        
    loader = DataLoader(TensorDataset(X_train, y_tr), batch_size=256, shuffle=True)
    
    for _ in range(10):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
            
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_test)) > 0.5).int().numpy()
        return accuracy_score(y_test.numpy(), preds)

accs = [train_eval(s) for s in [42, 43, 44, 45, 46]]
shuf = train_eval(42, True)

res = {
    "seeds_accuracy": accs,
    "mean_accuracy": float(np.mean(accs)),
    "std_accuracy": float(np.std(accs)),
    "shuffled_baseline_accuracy": shuf
}
os.makedirs("results/v3", exist_ok=True)
with open("results/v3/cnn_distance_multiseed.json", "w") as f:
    json.dump(res, f, indent=4)
