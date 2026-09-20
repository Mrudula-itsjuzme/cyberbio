"""
run_generative_benchmark.py — Phase 5 generative attack benchmark
Runs Generative (trained) and Generative-Untrained against the frozen N=30 source
manifests for both Bio-Cyber and Materials domains.
"""
import json, os, sys, math, time, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent / "materials-adversarial" / "src"))

from generative.model import DenoisingSeqAE
from materials_adv.models.transformer import TransformerRegressorModel
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PROPOSALS_FACTOR = 10

# ── Bio-Cyber target model (embedding-based CNN_Distance from run_benchmark.py) ──
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

# ── Proposer that wraps the denoising autoencoder ──
class GenerativeProposerLocal:
    """Lightweight proposer — no external util dependencies."""
    def __init__(self, domain, ckpt_path, config_path, max_len, trained=True):
        config = json.load(open(config_path))
        self.vocab_size = 4 if domain == "bio_cyber" else 36
        self.pad_idx = 0
        self.max_len = max_len
        self.noise_rate = config.get("noise_rate", 0.15)
        self.model = DenoisingSeqAE(
            vocab_size=self.vocab_size, pad_idx=self.pad_idx, **config["model_kwargs"]
        )
        if trained and Path(ckpt_path).exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        self.model.to(DEVICE); self.model.eval()
        self.valid_tokens = list(range(1, self.vocab_size + 1))

    def _corrupt(self, seq, rng):
        res = list(seq)
        for i in range(len(res)):
            if rng.random() < self.noise_rate:
                res[i] = int(rng.choice(self.valid_tokens))
        return res

    def propose(self, source_seq, rng, temperature=1.0):
        corrupted = self._corrupt(source_seq, rng)
        pad_len = max(0, self.max_len - len(corrupted))
        corr_pad = (corrupted + [self.pad_idx] * pad_len)[:self.max_len]
        src_t = torch.tensor([corr_pad], dtype=torch.long, device=DEVICE)
        trng = torch.Generator(device="cpu")
        trng.manual_seed(int(rng.integers(0, 2**31)))
        generated = self.model.generate(
            src=src_t, max_len=self.max_len, temperature=temperature,
            greedy=False, rng=trng
        )
        if self.pad_idx in generated:
            generated = generated[:generated.index(self.pad_idx)]
        return generated, 1

# ── helpers ──
def wf_edit(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        p, dp[0] = dp[0], i
        for j in range(1, n + 1):
            t = dp[j]
            dp[j] = p if s1[i-1] == s2[j-1] else 1 + min(p, dp[j], dp[j-1])
            p = t
    return dp[n]

def eval_bio(model, seq_ids, max_len=150):
    seq_ids = seq_ids[:max_len]
    pad = [0] * (max_len - len(seq_ids))
    t = torch.tensor([seq_ids + pad], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(t)).item()

def eval_mat(model, seq_ids, max_len=256):
    seq_ids = seq_ids[:max_len]
    pad = [0] * (max_len - len(seq_ids))
    t = torch.tensor([seq_ids + pad], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return model(t).item()

# ═══════════════════════════════════════════════════════════════════════
# BIO-CYBER
# ═══════════════════════════════════════════════════════════════════════
def run_bio_generative(untrained: bool = False):
    out_dir = ROOT / "results/generative/bio_cyber"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_model = CNN_Distance()
    bc_path = ROOT.parent.parent / "bio-cyber-adversarial/results/v3/models/cnn_clean/model.pt"
    target_model.load_state_dict(torch.load(bc_path, map_location=DEVICE, weights_only=True))
    target_model.to(DEVICE); target_model.eval()

    sources = json.load(open(ROOT / "results/manifests/bio_cyber_sources_n30.json"))
    bio_df = pd.read_csv(ROOT.parent.parent / "bio-cyber-adversarial/data/v3/dataset.csv")
    id2seq = dict(zip(bio_df["id"], bio_df["sequence"]))

    ckpt = str(ROOT / "results/generative/checkpoints/bio_cyber/checkpoint.pt")
    cfg = str(ROOT / "configs/generative_bio_cyber.json")
    proposer = GenerativeProposerLocal("bio_cyber", ckpt, cfg, 150, trained=not untrained)

    budgets = [5, 20, 50] if not untrained else [20]
    cond_name = "Generative" if not untrained else "Generative-Untrained"
    vocab_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    idx2char = {v: k for k, v in vocab_map.items()}

    results = []
    for si, row in enumerate(sources):
        src_seq_str = id2seq[row["source_id"]]
        src_seq = [vocab_map[c] for c in src_seq_str if c in vocab_map]
        orig_prob = eval_bio(target_model, src_seq)
        target_label = 1 - row["label"]

        for budget in budgets:
            seed_str = f"{row['source_id']}_{cond_name}_{budget}_42"
            rng = np.random.default_rng(
                int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            )
            start_time = time.time()
            max_prop = budget * MAX_PROPOSALS_FACTOR
            attack_q = prop_attempts = valid_cands = gen_inf = 0
            best_seq, best_drift = src_seq, 0.0

            while attack_q < budget and prop_attempts < max_prop:
                prop_attempts += 1
                cand, inf = proposer.propose(src_seq, rng)
                gen_inf += inf
                if len(cand) == 0:
                    cand = [1]
                valid_cands += 1
                cand_prob = eval_bio(target_model, cand)
                attack_q += 1
                drift = abs(cand_prob - orig_prob)
                if drift > best_drift:
                    best_drift = drift; best_seq = cand

            term = "QUERY_BUDGET_EXHAUSTED" if attack_q >= budget else \
                   "PROPOSAL_CAP_REACHED" if prop_attempts >= max_prop else \
                   "NO_VALID_CANDIDATE"
            final_prob = eval_bio(target_model, best_seq)
            succ = int((target_label == 1 and final_prob > 0.5) or
                       (target_label == 0 and final_prob < 0.5))
            results.append({
                "source_id": row["source_id"],
                "attack_condition": cond_name,
                "search_budget": budget,
                "success": succ,
                "original_prob": orig_prob,
                "final_prob": final_prob,
                "absolute_drift": best_drift,
                "edit_distance": wf_edit(src_seq, best_seq),
                "proposal_attempts": prop_attempts,
                "valid_candidates": valid_cands,
                "attack_model_queries": attack_q,
                "attribution_queries": 0,
                "total_model_queries": attack_q,
                "generator_inference_count": gen_inf,
                "runtime_seconds": time.time() - start_time,
                "termination_reason": term,
                "source_input": "".join(idx2char.get(c, "?") for c in src_seq),
                "candidate_input": "".join(idx2char.get(c, "?") for c in best_seq),
            })
        if (si + 1) % 10 == 0:
            print(f"  [{cond_name}] {si+1}/{len(sources)} sources done")

    tag = "untrained" if untrained else "trained"
    pd.DataFrame(results).to_csv(out_dir / f"per_example_results_{tag}.csv", index=False)
    print(f"[bio_cyber] {cond_name}: wrote {len(results)} rows")

# ═══════════════════════════════════════════════════════════════════════
# MATERIALS
# ═══════════════════════════════════════════════════════════════════════
def run_materials_generative(untrained: bool = False):
    out_dir = ROOT / "results/generative/materials"
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.load(open(ROOT.parent.parent / "materials-adversarial/data/processed/vocab.json"))
    char2idx = {c: i+1 for i, c in enumerate(vocab)}
    idx2char = {v: k for k, v in char2idx.items()}

    target_model = TransformerRegressorModel(
        vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4,
        dim_feedforward=128, dropout=0.1, max_seq_len=256
    )
    target_model.load_state_dict(torch.load(
        ROOT.parent.parent / "materials-adversarial/results/models/transformer_regressor/model.pt",
        map_location=DEVICE, weights_only=False
    ))
    target_model.to(DEVICE); target_model.eval()

    sources = json.load(open(ROOT / "results/manifests/materials_sources_n30.json"))

    ckpt = str(ROOT / "results/generative/checkpoints/materials/checkpoint.pt")
    cfg = str(ROOT / "configs/generative_materials.json")
    proposer = GenerativeProposerLocal("materials", ckpt, cfg, 256, trained=not untrained)

    budgets = [5, 20, 50] if not untrained else [20]
    cond_name = "Generative" if not untrained else "Generative-Untrained"

    results = []
    for si, row in enumerate(sources):
        src_str = row["SMILES"]
        src_seq = [char2idx[c] for c in src_str if c in char2idx]
        orig_tg = eval_mat(target_model, src_seq)
        src_mol = Chem.MolFromSmiles(src_str)
        src_fp = AllChem.GetMorganFingerprintAsBitVect(src_mol, 2, nBits=2048) if src_mol else None

        for budget in budgets:
            seed_str = f"{row['source_id']}_{cond_name}_{budget}_42"
            rng = np.random.default_rng(
                int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            )
            start_time = time.time()
            max_prop = budget * MAX_PROPOSALS_FACTOR
            attack_q = prop_attempts = valid_cands = gen_inf = 0
            best_seq, best_drift, best_str = src_seq, 0.0, src_str

            while attack_q < budget and prop_attempts < max_prop:
                prop_attempts += 1
                cand, inf = proposer.propose(src_seq, rng)
                gen_inf += inf
                cand_str = "".join(idx2char.get(c, "") for c in cand)
                mol = Chem.MolFromSmiles(cand_str)
                if mol is None: continue
                try:
                    canon = Chem.MolToSmiles(mol, canonical=True)
                except:
                    continue
                if not all(c in char2idx for c in canon): continue
                valid_cands += 1
                cand_seq = [char2idx[c] for c in canon]
                cand_tg = eval_mat(target_model, cand_seq)
                attack_q += 1
                drift = abs(cand_tg - orig_tg)
                if drift > best_drift:
                    best_drift = drift; best_seq = cand_seq; best_str = canon

            term = "QUERY_BUDGET_EXHAUSTED" if attack_q >= budget else \
                   "PROPOSAL_CAP_REACHED" if prop_attempts >= max_prop else \
                   "NO_VALID_CANDIDATE"
            final_tg = eval_mat(target_model, best_seq)
            best_mol = Chem.MolFromSmiles(best_str)
            tanimoto = np.nan
            if best_mol and src_fp:
                cand_fp = AllChem.GetMorganFingerprintAsBitVect(best_mol, 2, nBits=2048)
                tanimoto = DataStructs.TanimotoSimilarity(src_fp, cand_fp)
            results.append({
                "source_id": row["source_id"],
                "attack_condition": cond_name,
                "search_budget": budget,
                "success": np.nan,
                "original_prob": orig_tg,
                "final_prob": final_tg,
                "absolute_drift": best_drift,
                "edit_distance": wf_edit(src_str, best_str),
                "tanimoto_similarity": tanimoto,
                "proposal_attempts": prop_attempts,
                "valid_candidates": valid_cands,
                "attack_model_queries": attack_q,
                "attribution_queries": 0,
                "total_model_queries": attack_q,
                "generator_inference_count": gen_inf,
                "runtime_seconds": time.time() - start_time,
                "termination_reason": term,
                "source_input": src_str,
                "candidate_input": best_str,
                "selected_candidate_rdkit_valid": True,
                "selected_candidate_canonicalization_valid": True,
                "selected_candidate_tokenization_valid": True,
            })
        if (si + 1) % 10 == 0:
            print(f"  [{cond_name}] {si+1}/{len(sources)} sources done")

    tag = "untrained" if untrained else "trained"
    pd.DataFrame(results).to_csv(out_dir / f"per_example_results_{tag}.csv", index=False)
    print(f"[materials] {cond_name}: wrote {len(results)} rows")


if __name__ == "__main__":
    domain = sys.argv[1] if len(sys.argv) > 1 else "all"
    if domain in ("bio_cyber", "all"):
        run_bio_generative(untrained=False)
        run_bio_generative(untrained=True)
    if domain in ("materials", "all"):
        run_materials_generative(untrained=False)
        run_materials_generative(untrained=True)
