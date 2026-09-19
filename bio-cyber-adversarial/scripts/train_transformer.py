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
def encode(seq):
    return [vocab[c] for c in seq]

X_train = torch.tensor([encode(s) for s in df[df["split"]=="train"]["sequence"]], dtype=torch.long)
y_train = torch.tensor(df[df["split"]=="train"]["label"].values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor([encode(s) for s in df[df["split"]=="test"]["sequence"]], dtype=torch.long)
y_test = torch.tensor(df[df["split"]=="test"]["label"].values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

class CNN_Distance(nn.Module):
    # A CNN that flattens so it preserves spatial info!
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

model = CNN_Distance()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("Training CNN_Distance...")
for ep in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {ep} Loss: {total_loss/len(train_loader)}")

model.eval()
with torch.no_grad():
    out = model(X_test)
    preds = (torch.sigmoid(out) > 0.5).int().numpy()
    acc = accuracy_score(y_test.numpy(), preds)
    print(f"Test Accuracy: {acc}")

os.makedirs("results/v3/models/cnn_distance", exist_ok=True)
torch.save(model.state_dict(), "results/v3/models/cnn_distance/model.pt")
