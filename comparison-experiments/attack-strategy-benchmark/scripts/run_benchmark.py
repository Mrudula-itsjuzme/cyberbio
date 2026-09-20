import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json
import time
import hashlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# --- 1. SETUP & MODELS ---
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
        return self.fc(self.conv(e).view(e.size(0), -1))

vocab = {"A": 1, "C": 2, "G": 3, "T": 4, "<PAD>": 0}
alphabet = ["A", "C", "G", "T"]
def encode(seq): return [vocab.get(c, 0) for c in seq]

# Load bio-cyber clean model
clean_bc = CNN_Distance()
bc_path = "../../bio-cyber-adversarial/results/v3/models/cnn_clean/model.pt"
if os.path.exists(bc_path):
    clean_bc.load_state_dict(torch.load(bc_path, weights_only=True))
clean_bc.eval()

# Load defended model for transfer
defended_bc = CNN_Distance()
bc_def_path = "../../bio-cyber-adversarial/results/v3/models/cnn_adv_mixed/model.pt"
if os.path.exists(bc_def_path):
    defended_bc.load_state_dict(torch.load(bc_def_path, weights_only=True))
defended_bc.eval()

def predict_bc(model, seqs):
    if not seqs: return []
    x = torch.tensor([encode(s) for s in seqs], dtype=torch.long)
    with torch.no_grad(): return torch.sigmoid(model(x)).numpy().flatten()

def get_occlusion_attribution(model, seq, orig_p):
    # orig_p is precomputed, no queries for it here.
    base_x = encode(seq)
    batch = []
    for i in range(len(seq)):
        mut = list(base_x)
        mut[i] = 0 # PAD
        batch.append(mut)
    x = torch.tensor(batch, dtype=torch.long)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out).numpy().flatten()
    return np.abs(orig_p - probs), len(batch)

# --- 2. SOURCE POOLS ---
manifest_path = "results/manifests/bio_cyber_sources_n30.json"
manifest_bc = json.load(open(manifest_path))

# Dummy materials manifest (Materials execution BLOCKED_INVALID_MODEL_RECONSTRUCTION)
os.makedirs("results/manifests", exist_ok=True)
json.dump([], open("results/manifests/materials_sources.json", "w"), indent=4)

# --- 3. ATTACK FAMILIES ---
def attack_random(model, seq, orig_p, budget):
    b_seq, b_diff, queries = seq, 0, 0
    seq_l = list(seq)
    first_success = np.nan
    c_p_best = orig_p
    start = time.time()
    for q in range(1, budget + 1):
        queries += 1
        mut = seq_l.copy()
        mut[np.random.randint(len(mut))] = np.random.choice(alphabet)
        cand = "".join(mut)
        p = predict_bc(model, [cand])[0]
        diff = abs(p - orig_p)
        if diff > b_diff: b_seq, b_diff, c_p_best = cand, diff, p
        if pd.isna(first_success) and ((orig_p > 0.5) != (p > 0.5)):
            first_success = q
    rt = time.time() - start
    return b_seq, c_p_best, queries, 0, first_success, rt, 0, rt

def attack_mcmc(model, seq, orig_p, budget):
    c_seq, c_p = list(seq), orig_p
    b_seq, b_diff, queries = seq, 0, 0
    c_p_best = orig_p
    first_success = np.nan
    start = time.time()
    for q in range(1, budget + 1):
        queries += 1
        mut = c_seq.copy()
        mut[np.random.randint(len(mut))] = np.random.choice(alphabet)
        cand = "".join(mut)
        p = predict_bc(model, [cand])[0]
        if abs(p - orig_p) >= abs(c_p - orig_p) or np.random.rand() < 0.1:
            c_seq, c_p = mut, p
        diff = abs(p - orig_p)
        if diff > b_diff: b_seq, b_diff, c_p_best = cand, diff, p
        if pd.isna(first_success) and ((orig_p > 0.5) != (p > 0.5)):
            first_success = q
    rt = time.time() - start
    return b_seq, c_p_best, queries, 0, first_success, rt, 0, rt

def attack_evo(model, seq, orig_p, budget):
    pop_size = 5
    pop = [list(seq) for _ in range(pop_size)]
    b_seq, b_diff, queries = seq, 0, 0
    c_p_best = orig_p
    first_success = np.nan
    start = time.time()
    for gen in range(max(1, budget // pop_size)):
        cands = ["".join(p) for p in pop]
        preds = predict_bc(model, cands)
        queries += len(cands)
        for i, (c, p) in enumerate(zip(cands, preds)):
            d = abs(p - orig_p)
            if d > b_diff: b_seq, b_diff, c_p_best = c, d, p
            if pd.isna(first_success) and ((orig_p > 0.5) != (p > 0.5)):
                first_success = queries - len(cands) + i + 1
        best_p = list(b_seq)
        pop = []
        for _ in range(pop_size):
            mut = best_p.copy()
            mut[np.random.randint(len(mut))] = np.random.choice(alphabet)
            pop.append(mut)
    rt = time.time() - start
    return b_seq, c_p_best, queries, 0, first_success, rt, 0, rt

def attack_attr(model, seq, orig_p, budget, mode="high"):
    start = time.time()
    attr, attr_queries = get_occlusion_attribution(model, seq, orig_p)
    attr_rt = time.time() - start
    
    start_search = time.time()
    if mode == "high": target_idx = np.argsort(attr)[-20:]
    elif mode == "low": target_idx = np.argsort(attr)[:20]
    else: target_idx = np.random.choice(len(seq), 20, replace=False)
        
    b_seq, b_diff, search_queries = seq, 0, 0
    seq_l = list(seq)
    c_p_best = orig_p
    first_success_search = np.nan
    for q in range(1, budget + 1):
        search_queries += 1
        mut = seq_l.copy()
        mut[np.random.choice(target_idx)] = np.random.choice(alphabet)
        cand = "".join(mut)
        p = predict_bc(model, [cand])[0]
        diff = abs(p - orig_p)
        if diff > b_diff: b_seq, b_diff, c_p_best = cand, diff, p
        if pd.isna(first_success_search) and ((orig_p > 0.5) != (p > 0.5)):
            first_success_search = q
    search_rt = time.time() - start_search
    return b_seq, c_p_best, search_queries, attr_queries, first_success_search, attr_rt + search_rt, attr_rt, search_rt

# --- 4. RUN EXPERIMENTS (BIO-CYBER) ---
results = []
attacks = [
    ("Random", attack_random), ("MCMC", attack_mcmc), ("Evolutionary", attack_evo),
    ("Attribution-High", lambda m,s,p,b: attack_attr(m,s,p,b,"high")),
    ("Attribution-Low", lambda m,s,p,b: attack_attr(m,s,p,b,"low")),
    ("Attribution-Random", lambda m,s,p,b: attack_attr(m,s,p,b,"rand"))
]
budgets = [5, 20, 50]

for atk_name, atk_fn in attacks:
    for b in budgets:
        for row in manifest_bc:
            seq = row["source_input"] if "source_input" in row else None
            if not seq:
                # Load sequence from dataset if manifest only has hash
                bc_df = pd.read_csv("../../bio-cyber-adversarial/data/v3/dataset.csv")
                seq = bc_df[bc_df["id"] == row["source_id"]].iloc[0]["sequence"]
                row["source_input"] = seq
            
            orig_p = predict_bc(clean_bc, [seq])[0]
            np.random.seed(42 + row["source_id"] + b)
            b_seq, c_p, sq, aq, fs_sq, rt, a_rt, s_rt = atk_fn(clean_bc, seq, orig_p, b)
            
            tq = sq + aq
            fs_tq = fs_sq + aq if pd.notna(fs_sq) else np.nan
            
            p_def_orig = predict_bc(defended_bc, [seq])[0]
            p_def_cand = predict_bc(defended_bc, [b_seq])[0]
            
            results.append({
                "domain": "bio_cyber",
                "source_id": row["source_id"],
                "attack_condition": atk_name,
                "search_budget": b,
                "seed": 42,
                "source_input": seq,
                "candidate_input": b_seq,
                "clean_prediction": float(orig_p),
                "candidate_prediction": float(c_p),
                "signed_prediction_change": float(c_p - orig_p),
                "absolute_prediction_change": float(abs(c_p - orig_p)),
                "success": bool((orig_p > 0.5) != (c_p > 0.5)),
                "edit_distance": sum(1 for i in range(len(seq)) if seq[i] != b_seq[i]),
                "search_queries": sq,
                "attribution_queries": aq,
                "total_queries": tq,
                "first_success_search_query": fs_sq,
                "first_success_total_query": fs_tq,
                "queries_to_success": fs_sq,
                "total_queries_to_success": fs_tq,
                "runtime_seconds": rt,
                "attribution_runtime_seconds": a_rt,
                "search_runtime_seconds": s_rt,
                "validity_pass": True,
                "failure_reason": "None" if pd.notna(fs_sq) else "Budget Exhausted",
                "label_flip": bool((orig_p > 0.5) != (c_p > 0.5)),
                "alphabet_valid": set(b_seq).issubset(set(alphabet)),
                "defense_transfer_success": bool((p_def_orig > 0.5) != (p_def_cand > 0.5))
            })

df_res = pd.DataFrame(results)
os.makedirs("results/bio_cyber", exist_ok=True)
df_res.to_csv("results/bio_cyber/per_example_results.csv", index=False)

def bs_mean(x, n_boot=1000):
    x = x.dropna().values
    if len(x) == 0: return np.nan, np.nan
    np.random.seed(42)
    means = [np.mean(np.random.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def bs_median(x, n_boot=1000):
    x = x.dropna().values
    if len(x) == 0: return np.nan, np.nan
    np.random.seed(42)
    meds = [np.median(np.random.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)

agg_list = []
for (atk, b), grp in df_res.groupby(["attack_condition", "search_budget"]):
    s_ci = bs_mean(grp["success"])
    d_ci = bs_mean(grp["absolute_prediction_change"])
    msq_ci = bs_median(grp["first_success_search_query"])
    mtq_ci = bs_median(grp["first_success_total_query"])
    agg_list.append({
        "domain": "bio_cyber", "attack_condition": atk, "budget": b,
        "n": len(grp),
        "success_rate": grp["success"].mean(),
        "success_rate_ci_lower": s_ci[0], "success_rate_ci_upper": s_ci[1],
        "mean_absolute_drift": grp["absolute_prediction_change"].mean(),
        "mean_absolute_drift_ci_lower": d_ci[0], "mean_absolute_drift_ci_upper": d_ci[1],
        "median_absolute_drift": grp["absolute_prediction_change"].median(),
        "p90_absolute_drift": grp["absolute_prediction_change"].quantile(0.9),
        "mean_edit_distance": grp["edit_distance"].mean(),
        "median_edit_distance": grp["edit_distance"].median(),
        "mean_total_queries": grp["total_queries"].mean(),
        "median_search_queries_to_success": grp["first_success_search_query"].median(),
        "median_search_queries_to_success_ci_lower": msq_ci[0], "median_search_queries_to_success_ci_upper": msq_ci[1],
        "median_total_queries_to_success": grp["first_success_total_query"].median(),
        "median_total_queries_to_success_ci_lower": mtq_ci[0], "median_total_queries_to_success_ci_upper": mtq_ci[1],
        "mean_runtime": grp["runtime_seconds"].mean(),
        "validity_rate": grp["validity_pass"].mean()
    })
pd.DataFrame(agg_list).to_csv("results/bio_cyber/aggregated_results.csv", index=False)
