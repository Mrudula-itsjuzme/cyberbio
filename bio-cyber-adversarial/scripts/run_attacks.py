import torch
import torch.nn as nn
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

model = CNN_Distance()
model.load_state_dict(torch.load("results/v3/models/cnn_distance/model.pt", weights_only=True))
model.eval()

vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
def encode(seq):
    return [vocab[c] for c in seq]
def predict(seqs):
    if len(seqs) == 0: return []
    x = torch.tensor([encode(s) for s in seqs], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
        return torch.sigmoid(out).numpy().flatten()

df = pd.read_csv("data/v3/dataset.csv")
sources = df[(df["split"]=="test")].sample(50, random_state=42)

alphabet = ["A", "C", "G", "T"]

def attack_random(seq, orig_pred, budget):
    best_seq = seq
    best_diff = 0
    seq_list = list(seq)
    for _ in range(budget):
        mut = seq_list.copy()
        idx = np.random.randint(len(mut))
        mut[idx] = np.random.choice(alphabet)
        cand = "".join(mut)
        p = predict([cand])[0]
        diff = abs(p - orig_pred)
        if diff > best_diff:
            best_diff = diff
            best_seq = cand
    return best_seq, predict([best_seq])[0]

def attack_mcmc(seq, orig_pred, budget):
    curr_seq = list(seq)
    curr_pred = predict([seq])[0]
    best_seq = seq
    best_diff = 0
    for _ in range(budget):
        mut = curr_seq.copy()
        idx = np.random.randint(len(mut))
        mut[idx] = np.random.choice(alphabet)
        cand = "".join(mut)
        p = predict([cand])[0]
        # Accept if it increases diff
        diff = abs(p - orig_pred)
        curr_diff = abs(curr_pred - orig_pred)
        if diff >= curr_diff or np.random.rand() < 0.1:
            curr_seq = mut
            curr_pred = p
        if diff > best_diff:
            best_diff = diff
            best_seq = cand
    return best_seq, predict([best_seq])[0]

def attack_evo(seq, orig_pred, budget):
    # simple pop
    pop = [list(seq) for _ in range(5)]
    best_seq = seq
    best_diff = 0
    for _ in range(budget // 5):
        # eval pop
        cands = ["".join(p) for p in pop]
        preds = predict(cands)
        diffs = [abs(p - orig_pred) for p in preds]
        # update best
        for c, d in zip(cands, diffs):
            if d > best_diff:
                best_diff = d
                best_seq = c
        # mutate best to form new pop
        best_p = list(best_seq)
        new_pop = []
        for _ in range(5):
            mut = best_p.copy()
            idx = np.random.randint(len(mut))
            mut[idx] = np.random.choice(alphabet)
            new_pop.append(mut)
        pop = new_pop
    return best_seq, predict([best_seq])[0]

results = []
os.makedirs("results/v3/frozen_attack_banks", exist_ok=True)

import hashlib

for attack_name, attack_fn in [("random", attack_random), ("mcmc", attack_mcmc), ("evolutionary", attack_evo)]:
    bank = []
    for idx, row in sources.iterrows():
        seq = row["sequence"]
        orig_pred = predict([seq])[0]
        label = row["label"]
        
        for budget in [5, 20, 50]:
            np.random.seed(42 + idx + budget)
            cand, c_pred = attack_fn(seq, orig_pred, budget)
            
            diff = abs(c_pred - orig_pred)
            is_flip = (orig_pred > 0.5) != (c_pred > 0.5)
            edits = sum(1 for a, b in zip(seq, cand) if a != b)
            
            res = {
                "source_id": row["id"],
                "attack": attack_name,
                "budget": budget,
                "source_sequence": seq,
                "candidate_sequence": cand,
                "source_prediction": float(orig_pred),
                "candidate_prediction": float(c_pred),
                "absolute_prediction_change": float(diff),
                "label_flip": bool(is_flip),
                "number_of_edits": edits,
                "query_count": budget
            }
            results.append(res)
            if budget == 50:
                bank.append(res)
                
    # save frozen bank
    with open(f"results/v3/frozen_attack_banks/{attack_name}.jsonl", "w") as f:
        for b in bank:
            f.write(json.dumps(b) + "\n")
            
pd.DataFrame(results).to_csv("results/v3/attack_results.csv", index=False)
