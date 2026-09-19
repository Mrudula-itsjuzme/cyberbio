import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import json

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

df = pd.read_csv("data/v3/dataset.csv")

vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
def encode(seq):
    return [vocab[c] for c in seq]

# Adversarial training: randomly mutate 2% of positions
def augment(x_tensor):
    x_aug = x_tensor.clone()
    mask = torch.rand(x_aug.shape) < 0.02
    random_tokens = torch.randint(1, 5, x_aug.shape)
    x_aug[mask] = random_tokens[mask]
    return x_aug

X_train = torch.tensor([encode(s) for s in df[df["split"]=="train"]["sequence"]], dtype=torch.long)
y_train = torch.tensor(df[df["split"]=="train"]["label"].values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor([encode(s) for s in df[df["split"]=="test"]["sequence"]], dtype=torch.long)
y_test = torch.tensor(df[df["split"]=="test"]["label"].values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

model = CNN_Distance()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("Training CNN_Distance_Defended...")
for ep in range(10):
    model.train()
    for x, y in train_loader:
        # train on clean
        optimizer.zero_grad()
        out = model(x)
        loss1 = criterion(out, y)
        
        # train on augmented
        x_aug = augment(x)
        out_aug = model(x_aug)
        loss2 = criterion(out_aug, y)
        
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

os.makedirs("results/v3/models/cnn_defended", exist_ok=True)
torch.save(model.state_dict(), "results/v3/models/cnn_defended/model.pt")
print("Saved cnn_defended")
