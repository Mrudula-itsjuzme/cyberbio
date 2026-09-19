import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data
df = pd.read_csv("data/v3/dataset.csv")

vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
def encode(seq):
    return [vocab[c] for c in seq]

X_train = torch.tensor([encode(s) for s in df[df["split"]=="train"]["sequence"]], dtype=torch.long)
y_train = torch.tensor(df[df["split"]=="train"]["label"].values, dtype=torch.float32).unsqueeze(1)

X_val = torch.tensor([encode(s) for s in df[df["split"]=="val"]["sequence"]], dtype=torch.long)
y_val = torch.tensor(df[df["split"]=="val"]["label"].values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor([encode(s) for s in df[df["split"]=="test"]["sequence"]], dtype=torch.long)
y_test = torch.tensor(df[df["split"]=="test"]["label"].values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 16, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(150*16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        e = self.emb(x).view(x.size(0), -1)
        return self.fc(e)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 32, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        c = self.conv(e).squeeze(-1)
        return self.fc(c)

class SeqTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 64, padding_idx=0)
        self.pos = nn.Embedding(150, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        e = self.emb(x) + self.pos(pos)
        out = self.enc(e)
        return self.fc(out.mean(dim=1))

def train_model(model, name, epochs=15):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val = float('inf')
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x), y).item()
        
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(f"results/v3/models/{name}", exist_ok=True)
            torch.save(model.state_dict(), f"results/v3/models/{name}/model.pt")

    model.load_state_dict(torch.load(f"results/v3/models/{name}/model.pt", weights_only=True))
    model.eval()
    preds, probs, true = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            pr = torch.sigmoid(out).numpy()
            probs.extend(pr)
            preds.extend((pr > 0.5).astype(int))
            true.extend(y.numpy())
            
    res = {
        "accuracy": accuracy_score(true, preds),
        "balanced_accuracy": balanced_accuracy_score(true, preds),
        "precision": precision_score(true, preds, zero_division=0),
        "recall": recall_score(true, preds, zero_division=0),
        "f1": f1_score(true, preds, zero_division=0),
        "auroc": roc_auc_score(true, probs)
    }
    with open(f"results/v3/models/{name}/metrics.json", "w") as f:
        json.dump(res, f, indent=4)
    print(f"{name} test accuracy: {res['accuracy']:.4f}")
    return model

torch.manual_seed(42)
np.random.seed(42)

mlp = train_model(MLP(), "mlp_baseline")
cnn = train_model(CNN(), "cnn_sequence")
tf = train_model(SeqTransformer(), "transformer_sequence")
