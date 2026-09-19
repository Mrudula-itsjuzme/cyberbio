import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

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

model = CNN_Distance()
model.load_state_dict(torch.load("results/v3/models/cnn_distance/model.pt", weights_only=True))
model.eval()

vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
def encode(seq):
    return [vocab[c] for c in seq]
def predict(seqs):
    x = torch.tensor([encode(s) for s in seqs], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
        return torch.sigmoid(out).numpy().flatten()

df = pd.read_csv("data/v3/dataset.csv")
test_c1 = df[(df["split"]=="test") & (df["label"]==1)].head(200)

results = []
for idx, row in test_c1.iterrows():
    seq = row["sequence"]
    p1, p2 = row["pos1"], row["pos2"]
    dist = row["distance"]
    order = row["order"]
    
    if order == "A_then_B":
        m1, m2 = "ATGC", "GCAT"
    else:
        m1, m2 = "GCAT", "ATGC"
        
    orig_prob = predict([seq])[0]
    
    # 1. Alter spacing (dist=65)
    bg = list(seq)
    bg[p1:p1+4] = ["A"]*4
    bg[p2:p2+4] = ["A"]*4
    new_p1 = min(p1, 150 - 65 - 4)
    new_p2 = new_p1 + 65
    c1 = list("".join(bg))
    c1[new_p1:new_p1+4] = list(m1)
    c1[new_p2:new_p2+4] = list(m2)
    p_spacing = predict(["".join(c1)])[0]
    
    # 2. Swap motif order
    c2 = list(seq)
    c2[p1:p1+4] = list(m2)
    c2[p2:p2+4] = list(m1)
    p_order = predict(["".join(c2)])[0]
    
    # 3. Move both together (shift by 10)
    c3 = list("".join(bg))
    shift_p1 = (p1 + 10) % (150 - dist - 4)
    shift_p2 = shift_p1 + dist
    c3[shift_p1:shift_p1+4] = list(m1)
    c3[shift_p2:shift_p2+4] = list(m2)
    p_move = predict(["".join(c3)])[0]
    
    # 4. Randomize background
    c4 = list("".join(np.random.choice(["A", "C", "G", "T"], size=150)))
    c4[p1:p1+4] = list(m1)
    c4[p2:p2+4] = list(m2)
    p_bg = predict(["".join(c4)])[0]
    
    # 5. Remove Motif A
    c5 = list(seq)
    if order == "A_then_B":
        c5[p1:p1+4] = ["A", "A", "A", "A"]
    else:
        c5[p2:p2+4] = ["A", "A", "A", "A"]
    p_rmA = predict(["".join(c5)])[0]
    
    # 6. Remove Motif B
    c6 = list(seq)
    if order == "A_then_B":
        c6[p2:p2+4] = ["A", "A", "A", "A"]
    else:
        c6[p1:p1+4] = ["A", "A", "A", "A"]
    p_rmB = predict(["".join(c6)])[0]
    
    results.append({
        "orig_prob": float(orig_prob),
        "spacing_prob": float(p_spacing),
        "order_prob": float(p_order),
        "move_prob": float(p_move),
        "bg_prob": float(p_bg),
        "rmA_prob": float(p_rmA),
        "rmB_prob": float(p_rmB)
    })

res_df = pd.DataFrame(results)
res_df.to_csv("results/v3/relationship_counterfactuals.csv", index=False)
print("Mean probabilities under counterfactuals (Class 1 -> closer to 1 is better):")
print(res_df.mean())
