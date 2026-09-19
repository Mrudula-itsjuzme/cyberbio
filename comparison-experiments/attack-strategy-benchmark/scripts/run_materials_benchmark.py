import sys, json, os, time, math, hashlib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs

sys.path.insert(0, '../../materials-adversarial/src')
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.data.tokenizer import tokenize
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.deletion import DeletionAttack
def edit_distance(s1, s2):
    """Wagner-Fischer Levenshtein distance (pure Python, no external dep)."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]

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
    for i in range(len(tokens)):
        mut = pred.encode(seq)
        mut[i] = 0 # PAD
        batch.append(mut)
    x = torch.tensor(batch, dtype=torch.long)
    with torch.no_grad():
        out = pred.model(x).squeeze(-1).numpy()
    return np.abs(orig_p - out), len(tokens), tokens

# --- 2. SOURCE POOLS ---
manifest = json.load(open("../results/manifests/materials_sources_n30.json"))

def get_similarity(s1, s2):
    m1 = Chem.MolFromSmiles(s1)
    m2 = Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return np.nan
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m1, 2)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(m2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def check_validity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, False, ""
    try:
        can = Chem.MolToSmiles(mol, canonical=True)
        return True, True, can
    except:
        return True, False, ""

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

def mutate_once(tokens, rng):
    mutators = [
        SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1),
        InsertionAttack(rng, allowed_tokens=vocab, attack_budget=1),
        DeletionAttack(rng, attack_budget=1)
    ]
    m = rng.choice(mutators)
    out = m.generate(tokens, n_variants=1)
    if out:
        return out[0].adversarial_representation, list(out[0].adversarial_tokens), 1
    return "".join(tokens), list(tokens), 1

def run_attack(attack_fn, seq, orig_p, budget, rng, **kwargs):
    MAX_PROPOSALS = budget * 10
    start = time.time()
    res = attack_fn(seq, orig_p, budget, MAX_PROPOSALS, rng, **kwargs)
    rt = time.time() - start
    return res + (rt,)

def attack_random(seq, orig_p, budget, max_proposals, rng):
    c_tokens = tokenize(seq)
    b_seq, b_diff, c_p_best = seq, 0, orig_p
    proposals, valid_cands, queries = 0, 0, 0
    rdk_val, can_val, tok_val = 0, 0, 0
    term_reason = "QUERY_BUDGET_EXHAUSTED"
    
    while queries < budget and proposals < max_proposals:
        cand, cand_t, att = mutate_once(c_tokens, rng)
        proposals += att
        
        rdk, can, can_smiles = check_validity(cand)
        tok = check_tokenization(cand)
        if rdk: rdk_val += 1
        if can: can_val += 1
        if tok: tok_val += 1
        
        if rdk and can and tok:
            valid_cands += 1
            queries += 1
            p = pred.predict([cand])[0]
            diff = abs(p - orig_p)
            if diff > b_diff:
                b_seq, b_diff, c_p_best = cand, diff, p
                
        if queries >= budget: break
        
    if queries < budget and proposals >= max_proposals:
        term_reason = "PROPOSAL_CAP_REACHED"
        if valid_cands == 0: term_reason = "NO_VALID_CANDIDATE"
        
    return b_seq, c_p_best, proposals, valid_cands, queries, 0, term_reason, rdk_val, can_val, tok_val

def attack_mcmc(seq, orig_p, budget, max_proposals, rng):
    c_tokens = tokenize(seq)
    c_seq, c_p = seq, orig_p
    b_seq, b_diff, c_p_best = seq, 0, orig_p
    proposals, valid_cands, queries = 0, 0, 0
    rdk_val, can_val, tok_val = 0, 0, 0
    term_reason = "QUERY_BUDGET_EXHAUSTED"
    
    while queries < budget and proposals < max_proposals:
        cand, cand_t, att = mutate_once(c_tokens, rng)
        proposals += att
        
        rdk, can, can_smiles = check_validity(cand)
        tok = check_tokenization(cand)
        if rdk: rdk_val += 1
        if can: can_val += 1
        if tok: tok_val += 1
        
        if rdk and can and tok:
            valid_cands += 1
            queries += 1
            p = pred.predict([cand])[0]
            diff = abs(p - orig_p)
            
            if diff > b_diff:
                b_seq, b_diff, c_p_best = cand, diff, p
            
            delta = abs(p - orig_p) - abs(c_p - orig_p)
            if delta > 0 or rng.random() < math.exp(delta / 10.0):
                c_seq, c_p, c_tokens = cand, p, cand_t
                
    if queries < budget and proposals >= max_proposals:
        term_reason = "PROPOSAL_CAP_REACHED"
        if valid_cands == 0: term_reason = "NO_VALID_CANDIDATE"
        
    return b_seq, c_p_best, proposals, valid_cands, queries, 0, term_reason, rdk_val, can_val, tok_val

def attack_evo(seq, orig_p, budget, max_proposals, rng):
    pop = [(tokenize(seq), seq, orig_p)]
    b_seq, b_diff, c_p_best = seq, 0, orig_p
    proposals, valid_cands, queries = 0, 0, 0
    rdk_val, can_val, tok_val = 0, 0, 0
    term_reason = "QUERY_BUDGET_EXHAUSTED"
    
    while queries < budget and proposals < max_proposals:
        new_pop = []
        for t, s, p in pop:
            for _ in range(5):
                if queries >= budget or proposals >= max_proposals: break
                
                cand, cand_t, att = mutate_once(t, rng)
                proposals += att
                
                rdk, can, can_smiles = check_validity(cand)
                tok = check_tokenization(cand)
                if rdk: rdk_val += 1
                if can: can_val += 1
                if tok: tok_val += 1
                
                if rdk and can and tok:
                    valid_cands += 1
                    queries += 1
                    cp = pred.predict([cand])[0]
                    diff = abs(cp - orig_p)
                    new_pop.append((cand_t, cand, cp, diff))
                    if diff > b_diff:
                        b_seq, b_diff, c_p_best = cand, diff, cp
                        
        if not new_pop:
            continue # try again next generation if budget permits
            
        new_pop.sort(key=lambda x: x[3], reverse=True)
        pop = [(x[0], x[1], x[2]) for x in new_pop[:3]]
        
    if queries < budget and proposals >= max_proposals:
        term_reason = "PROPOSAL_CAP_REACHED"
        if valid_cands == 0: term_reason = "NO_VALID_CANDIDATE"
        
    return b_seq, c_p_best, proposals, valid_cands, queries, 0, term_reason, rdk_val, can_val, tok_val

def mutate_pos(tokens, pos, rng):
    c = tokens.copy()
    if rng.random() < 0.5:
        c[pos] = rng.choice(vocab)
    else:
        del c[pos]
    return "".join(c), c, 1

def attack_attribution(seq, orig_p, budget, max_proposals, rng, attr_mode):
    attr_scores, attr_queries, tokens = get_occlusion_attribution(seq, orig_p)
    
    proposals, valid_cands, queries = 0, 0, 0
    rdk_val, can_val, tok_val = 0, 0, 0
    b_seq, b_diff, c_p_best = seq, 0, orig_p
    term_reason = "QUERY_BUDGET_EXHAUSTED"
    
    while queries < budget and proposals < max_proposals:
        if attr_mode == "High":
            pos = np.argmax(attr_scores)
        elif attr_mode == "Low":
            pos = np.argmin(attr_scores)
        else:
            pos = rng.integers(0, len(tokens))
            
        cand, cand_t, att = mutate_pos(tokens, pos, rng)
        proposals += att
        
        rdk, can, can_smiles = check_validity(cand)
        tok = check_tokenization(cand)
        if rdk: rdk_val += 1
        if can: can_val += 1
        if tok: tok_val += 1
        
        if rdk and can and tok:
            valid_cands += 1
            queries += 1
            p = pred.predict([cand])[0]
            diff = abs(p - orig_p)
            if diff > b_diff:
                b_seq, b_diff, c_p_best = cand, diff, p
                
    if queries < budget and proposals >= max_proposals:
        term_reason = "PROPOSAL_CAP_REACHED"
        if valid_cands == 0: term_reason = "NO_VALID_CANDIDATE"
        
    return b_seq, c_p_best, proposals, valid_cands, queries, attr_queries, term_reason, rdk_val, can_val, tok_val

# --- 4. EXECUTION ---
results = []
budgets = [5, 20, 50]
for row in manifest:
    seq = row["SMILES"]
    sid = row["source_id"]
    orig_p = pred.predict([seq])[0]
    
    for b in budgets:
        # We need a stable random seed string
        for cond, fn, kwargs in [
            ("Random", attack_random, {}),
            ("MCMC", attack_mcmc, {}),
            ("Evolutionary", attack_evo, {}),
            ("Attribution-High", attack_attribution, {"attr_mode": "High"}),
            ("Attribution-Low", attack_attribution, {"attr_mode": "Low"}),
            ("Attribution-Random", attack_attribution, {"attr_mode": "Random"})
        ]:
            seed_str = f"{sid}_{cond}_{b}_42"
            rng_seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(rng_seed)
            
            res = run_attack(fn, seq, orig_p, b, rng, **kwargs)
            results.append((sid, cond, b, res))

import csv
os.makedirs("../results/materials", exist_ok=True)
with open("../results/materials/per_example_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["domain", "source_id", "attack_condition", "search_budget", "seed",
                "source_input", "candidate_input", "target_Tg", "clean_prediction",
                "candidate_prediction", "signed_prediction_change", "absolute_prediction_change",
                "success", "edit_distance", "tanimoto_similarity", "fingerprint_type", 
                "proposal_attempts", "valid_candidates", "attack_model_queries",
                "attribution_queries", "total_model_queries", "termination_reason", "runtime_seconds",
                "rdkit_valid_proposals", "canonicalizable_proposals", "tokenizable_proposals",
                "selected_candidate_rdkit_valid", "selected_candidate_canonicalization_valid", 
                "selected_candidate_tokenization_valid",
                "source_canonical_smiles", "candidate_canonical_smiles"])
                
    for sid, cond, b, res in results:
        (b_seq, c_p_best, proposals, valid_cands, attack_queries, attr_queries, 
         term_reason, rdk_val, can_val, tok_val, rt) = res
         
        orig_seq = [r["SMILES"] for r in manifest if r["source_id"] == sid][0]
        tgt = [r["target_Tg"] for r in manifest if r["source_id"] == sid][0]
        orig_p = pred.predict([orig_seq])[0]
        
        diff = abs(c_p_best - orig_p)
        signed = c_p_best - orig_p
        ed = edit_distance(orig_seq, b_seq)
        sim = get_similarity(orig_seq, b_seq)
        
        _, _, s_can = check_validity(orig_seq)
        s_can = s_can if s_can else np.nan
        b_rdk, b_can, b_can_str = check_validity(b_seq)
        b_can_str = b_can_str if b_can_str else np.nan
        b_tok = check_tokenization(b_seq)
        
        w.writerow(["materials", sid, cond, b, 42, orig_seq, b_seq, tgt, orig_p,
                    c_p_best, signed, diff, "", ed, sim, "Morgan_R=2",
                    proposals, valid_cands, attack_queries, attr_queries, 
                    attack_queries + attr_queries, term_reason, rt,
                    rdk_val, can_val, tok_val,
                    b_rdk, b_can, b_tok, s_can, b_can_str])

print("Finished evaluating materials attacks.")
