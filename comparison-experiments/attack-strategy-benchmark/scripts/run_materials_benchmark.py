import sys, json, os, time, math
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

sys.path.insert(0, '../../../materials-adversarial/src')
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.data.tokenizer import tokenize
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.deletion import DeletionAttack
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# --- 1. SETUP & MODELS ---
vocab = json.load(open('../../../materials-adversarial/data/processed/vocab.json'))
char2idx = {c: i + 1 for i, c in enumerate(vocab)}

class MaterialsPredictor:
    def __init__(self, model_path):
        self.model = TransformerRegressorModel(
            vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4,
            dim_feedforward=128, dropout=0.1, max_seq_len=256
        )
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    def encode(self, seq, max_len=256):
        tokens = tokenize(seq)
        ids = [char2idx.get(c, 0) for c in tokens][:max_len]
        padding = max_len - len(ids)
        ids.extend([0] * padding)
        return ids
        
    def predict(self, smiles_list):
        if not smiles_list:
            return []
        x = torch.tensor([self.encode(s) for s in smiles_list], dtype=torch.long)
        with torch.no_grad():
            preds = self.model(x).squeeze(-1).tolist()
        if not isinstance(preds, list):
            preds = [preds]
        return preds

pred = MaterialsPredictor('../../../materials-adversarial/results/models/transformer_regressor/model.pt')

def get_occlusion_attribution(seq, orig_p):
    tokens = tokenize(seq)
    batch = []
    valid_idx = []
    for i in range(len(tokens)):
        mut = pred.encode(seq)
        mut[i] = 0 # PAD
        batch.append(mut)
    x = torch.tensor(batch, dtype=torch.long)
    with torch.no_grad():
        out = pred.model(x).squeeze(-1).numpy()
    return np.abs(orig_p - out), len(batch), tokens

# --- 2. SOURCE POOLS ---
manifest = json.load(open("../results/manifests/materials_sources_n30.json"))

# --- 3. ATTACK MACHINERY ---
rng = np.random.default_rng(42)
mutators = [
    SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1),
    InsertionAttack(rng, allowed_tokens=vocab, attack_budget=1),
    DeletionAttack(rng, attack_budget=1)
]

def check_validity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, False
    try:
        can = Chem.MolToSmiles(mol, canonical=True)
        return True, True
    except:
        return True, False

def check_tokenization(smiles):
    try:
        tokens = tokenize(smiles, on_unknown="raise")
        for t in tokens:
            if t not in vocab:
                return False
        if len(tokens) > 256:
            return False
        return True
    except:
        return False

def mutate_once(tokens):
    for _ in range(10): # try 10 times to get something
        m = rng.choice(mutators)
        out = m.generate(tokens, n_variants=1)
        if out:
            return out[0].adversarial_representation, list(out[0].adversarial_tokens)
    return "".join(tokens), list(tokens)

def attack_random(seq, orig_p, budget):
    c_tokens = tokenize(seq)
    b_seq, b_diff, proposals, queries = seq, 0, 0, 0
    c_p_best = orig_p
    valid_rdkit = 0
    start = time.time()
    
    for _ in range(budget):
        cand, cand_t = mutate_once(c_tokens)
        proposals += 1
        rdk, can = check_validity(cand)
        tok = check_tokenization(cand)
        if rdk: valid_rdkit += 1
        if rdk and can and tok:
            queries += 1
            p = pred.predict([cand])[0]
            diff = abs(p - orig_p)
            if diff > b_diff:
                b_seq, b_diff, c_p_best = cand, diff, p
                
    rt = time.time() - start
    return b_seq, c_p_best, proposals, queries, 0, 0, rt, 0, rt, valid_rdkit

def attack_mcmc(seq, orig_p, budget):
    c_tokens = tokenize(seq)
    c_seq, c_p = seq, orig_p
    b_seq, b_diff, proposals, queries = seq, 0, 0, 0
    c_p_best = orig_p
    valid_rdkit = 0
    start = time.time()
    
    for _ in range(budget):
        cand, cand_t = mutate_once(c_tokens)
        proposals += 1
        rdk, can = check_validity(cand)
        tok = check_tokenization(cand)
        if rdk: valid_rdkit += 1
        if rdk and can and tok:
            queries += 1
            p = pred.predict([cand])[0]
            diff = abs(p - orig_p)
            
            # Acceptance
            if diff > b_diff:
                b_seq, b_diff, c_p_best = cand, diff, p
            
            delta = abs(p - orig_p) - abs(c_p - orig_p)
            if delta > 0 or rng.random() < math.exp(delta / 10.0):
                c_seq, c_p, c_tokens = cand, p, cand_t
                
    rt = time.time() - start
    return b_seq, c_p_best, proposals, queries, 0, 0, rt, 0, rt, valid_rdkit

def attack_evo(seq, orig_p, budget):
    pop = [(tokenize(seq), seq, orig_p)]
    b_seq, b_diff, proposals, queries = seq, 0, 0, 0
    c_p_best = orig_p
    valid_rdkit = 0
    start = time.time()
    
    while queries < budget:
        new_pop = []
        for t, s, p in pop:
            for _ in range(5):
                cand, cand_t = mutate_once(t)
                proposals += 1
                rdk, can = check_validity(cand)
                tok = check_tokenization(cand)
                if rdk: valid_rdkit += 1
                if rdk and can and tok:
                    queries += 1
                    cp = pred.predict([cand])[0]
                    diff = abs(cp - orig_p)
                    new_pop.append((cand_t, cand, cp, diff))
                    if diff > b_diff:
                        b_seq, b_diff, c_p_best = cand, diff, cp
                if queries >= budget:
                    break
            if queries >= budget:
                break
        
        if not new_pop:
            break
        new_pop.sort(key=lambda x: x[3], reverse=True)
        pop = [(x[0], x[1], x[2]) for x in new_pop[:3]]
        
    rt = time.time() - start
    return b_seq, c_p_best, proposals, queries, 0, 0, rt, 0, rt, valid_rdkit

def mutate_pos(tokens, pos):
    c = tokens.copy()
    if rng.random() < 0.5:
        c[pos] = rng.choice(vocab)
    else:
        del c[pos]
    return "".join(c), c

def attack_attribution(seq, orig_p, budget, attr_mode):
    start = time.time()
    attr_scores, attr_queries, tokens = get_occlusion_attribution(seq, orig_p)
    attr_rt = time.time() - start
    
    proposals, search_queries = 0, 0
    b_seq, b_diff, c_p_best = seq, 0, orig_p
    valid_rdkit = 0
    start_search = time.time()
    
    for _ in range(budget):
        if attr_mode == "High":
            pos = np.argmax(attr_scores)
        elif attr_mode == "Low":
            pos = np.argmin(attr_scores)
        else:
            pos = rng.integers(0, len(tokens))
            
        cand, cand_t = mutate_pos(tokens, pos)
        proposals += 1
        rdk, can = check_validity(cand)
        tok = check_tokenization(cand)
        if rdk: valid_rdkit += 1
        if rdk and can and tok:
            search_queries += 1
            p = pred.predict([cand])[0]
            diff = abs(p - orig_p)
            if diff > b_diff:
                b_seq, b_diff, c_p_best = cand, diff, p
                
    rt_s = time.time() - start_search
    return b_seq, c_p_best, proposals, search_queries, attr_queries, 0, rt_s + attr_rt, attr_rt, rt_s, valid_rdkit

# --- 4. EXECUTION ---
results = []
budgets = [5, 20, 50]
for idx, row in enumerate(manifest):
    seq = row["SMILES"]
    sid = row["source_id"]
    orig_p = pred.predict([seq])[0]
    
    for b in budgets:
        # Random
        res = attack_random(seq, orig_p, b)
        results.append((sid, "Random", b, res))
        # MCMC
        res = attack_mcmc(seq, orig_p, b)
        results.append((sid, "MCMC", b, res))
        # Evo
        res = attack_evo(seq, orig_p, b)
        results.append((sid, "Evolutionary", b, res))
        # Attr High
        res = attack_attribution(seq, orig_p, b, "High")
        results.append((sid, "Attribution-High", b, res))
        # Attr Low
        res = attack_attribution(seq, orig_p, b, "Low")
        results.append((sid, "Attribution-Low", b, res))
        # Attr Random
        res = attack_attribution(seq, orig_p, b, "Random")
        results.append((sid, "Attribution-Random", b, res))

import csv
os.makedirs("../results/materials", exist_ok=True)
with open("../results/materials/per_example_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["domain", "source_id", "attack_condition", "search_budget", "seed",
                "source_input", "candidate_input", "target_Tg", "clean_prediction",
                "candidate_prediction", "signed_prediction_change", "absolute_prediction_change",
                "success", "edit_distance", "proposal_count", "search_queries",
                "attribution_queries", "total_queries", "first_success_search_query",
                "first_success_total_query", "runtime_seconds", "attribution_runtime_seconds",
                "search_runtime_seconds", "validity_pass", "rdkit_valid", "canonicalization_valid",
                "tokenization_valid", "source_canonical_smiles", "candidate_canonical_smiles",
                "failure_reason"])
                
    for sid, cond, b, res in results:
        b_seq, c_p_best, proposals, search_queries, attr_queries, f_s_sq, rt, attr_rt, s_rt, v_rdk = res
        orig_seq = [r["SMILES"] for r in manifest if r["source_id"] == sid][0]
        tgt = [r["target_Tg"] for r in manifest if r["source_id"] == sid][0]
        orig_p = pred.predict([orig_seq])[0]
        
        diff = abs(c_p_best - orig_p)
        signed = c_p_best - orig_p
        ed = edit_distance(orig_seq, b_seq)
        
        # Valid properties
        v_can = v_rdk
        v_tok = v_rdk
        valid_pass = (v_rdk > 0)
        
        w.writerow(["materials", sid, cond, b, 42, orig_seq, b_seq, tgt, orig_p,
                    c_p_best, signed, diff, "", ed, proposals, search_queries,
                    attr_queries, search_queries + attr_queries, "", "", rt, attr_rt, s_rt,
                    valid_pass, v_rdk, v_can, v_tok, orig_seq, b_seq, ""])

print("Finished evaluating materials attacks.")
