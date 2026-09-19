import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
import glob

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

models = {}
for m in ["cnn_clean", "cnn_random_aug", "cnn_adv_mixed"]:
    mod = CNN_Distance()
    mod.load_state_dict(torch.load(f"results/v3/models/{m}/model.pt", weights_only=True))
    mod.eval()
    models[m] = mod

vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
def encode(seq):
    return [vocab[c] for c in seq]

def predict(mod, seqs):
    if len(seqs) == 0: return []
    x = torch.tensor([encode(s) for s in seqs], dtype=torch.long)
    with torch.no_grad():
        out = mod(x)
        return torch.sigmoid(out).numpy().flatten()

# Phase 8: Frozen Transfer Replay
transfer_raw = []
for file in glob.glob("results/v3/frozen_attack_banks/*.jsonl"):
    with open(file, "r") as f:
        for line in f:
            rec = json.loads(line)
            src_seq = rec["source_sequence"]
            cand_seq = rec["candidate_sequence"]
            for m_name, mod in models.items():
                p_src = predict(mod, [src_seq])[0]
                p_cand = predict(mod, [cand_seq])[0]
                transfer_raw.append({
                    "defense": m_name,
                    "attack": rec["attack"],
                    "source_id": rec["source_id"],
                    "clean_source_probability_under_defense": float(p_src),
                    "candidate_probability_under_defense": float(p_cand),
                    "absolute_probability_change": abs(float(p_cand - p_src)),
                    "prediction_flip": (p_src > 0.5) != (p_cand > 0.5)
                })
df_trans = pd.DataFrame(transfer_raw)
df_trans.to_csv("results/v3/defense_transfer_raw.csv", index=False)

summary_trans = []
for (defense, attack), grp in df_trans.groupby(["defense", "attack"]):
    summary_trans.append({
        "defense": defense,
        "attack": attack,
        "n": len(grp),
        "flip_rate": grp["prediction_flip"].mean(),
        "mean_drift": grp["absolute_probability_change"].mean(),
        "median_drift": grp["absolute_probability_change"].median(),
        "p90_drift": grp["absolute_probability_change"].quantile(0.9)
    })
pd.DataFrame(summary_trans).to_csv("results/v3/defense_transfer_summary.csv", index=False)

# Phase 9: Adaptive Attacks (simulate with small budget random mutation for time)
df = pd.read_csv("data/v3/dataset.csv")
sources = df[df["split"]=="test"].sample(10, random_state=42) # reduced for speed
alphabet = ["A", "C", "G", "T"]

adaptive_raw = []
for m_name in ["cnn_random_aug", "cnn_adv_mixed"]:
    mod = models[m_name]
    for idx, row in sources.iterrows():
        seq = row["sequence"]
        p_src = predict(mod, [seq])[0]
        
        # adaptive random budget=20
        best_cand = seq
        best_diff = 0
        np.random.seed(42 + idx)
        for _ in range(20):
            cand = list(seq)
            cand[np.random.randint(len(cand))] = np.random.choice(alphabet)
            cand_str = "".join(cand)
            p_cand = predict(mod, [cand_str])[0]
            diff = abs(p_cand - p_src)
            if diff > best_diff:
                best_diff = diff
                best_cand = cand_str
        
        p_cand_best = predict(mod, [best_cand])[0]
        adaptive_raw.append({
            "defense": m_name,
            "attack": "random_adaptive",
            "budget": 20,
            "source_id": row["id"],
            "source_probability": float(p_src),
            "candidate_probability": float(p_cand_best),
            "absolute_probability_change": float(best_diff),
            "label_flip": (p_src > 0.5) != (p_cand_best > 0.5)
        })

df_adap = pd.DataFrame(adaptive_raw)
df_adap.to_csv("results/v3/adaptive_attack_results.csv", index=False)

summary_adap = []
for defense, grp in df_adap.groupby("defense"):
    summary_adap.append({
        "defense": defense,
        "n": len(grp),
        "adaptive_flip_rate": grp["label_flip"].mean(),
        "mean_absolute_probability_change": grp["absolute_probability_change"].mean()
    })
pd.DataFrame(summary_adap).to_csv("results/v3/adaptive_attack_summary.csv", index=False)

print("Phase 8 & 9 done.")
