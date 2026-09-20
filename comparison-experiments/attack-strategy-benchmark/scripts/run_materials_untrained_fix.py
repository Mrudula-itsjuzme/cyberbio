import json, os, sys, math, time, hashlib, signal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

class GenerativeProposerLocal:
    def __init__(self, domain, ckpt_path, config_path, max_len, trained=True):
        config = json.load(open(config_path))
        self.vocab_size = 36
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

def eval_mat(model, seq_ids, max_len=256):
    seq_ids = seq_ids[:max_len]
    pad = [0] * (max_len - len(seq_ids))
    t = torch.tensor([seq_ids + pad], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return model(t).item()

# Timeout handler for RDKit hangs
class TimeoutException(Exception): pass
def timeout_handler(signum, frame):
    raise TimeoutException()
signal.signal(signal.SIGALRM, timeout_handler)

def run_materials_untrained():
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

    cfg = str(ROOT / "configs/generative_materials.json")
    proposer = GenerativeProposerLocal("materials", "NO_CKPT", cfg, 256, trained=False)

    budgets = [20]
    cond_name = "Generative-Untrained"

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
                
                # Use alarm to prevent RDKit from hanging on garbage SMILES
                signal.alarm(1) 
                try:
                    mol = Chem.MolFromSmiles(cand_str)
                    if mol is not None:
                        canon = Chem.MolToSmiles(mol, canonical=True)
                        if all(c in char2idx for c in canon):
                            valid_cands += 1
                            cand_seq = [char2idx[c] for c in canon]
                            cand_tg = eval_mat(target_model, cand_seq)
                            attack_q += 1
                            drift = abs(cand_tg - orig_tg)
                            if drift > best_drift:
                                best_drift = drift; best_seq = cand_seq; best_str = canon
                except TimeoutException:
                    pass # Hang occurred, treat as invalid
                except Exception:
                    pass
                finally:
                    signal.alarm(0)

            term = "QUERY_BUDGET_EXHAUSTED" if attack_q >= budget else \
                   "PROPOSAL_CAP_REACHED" if prop_attempts >= max_prop else \
                   "NO_VALID_CANDIDATE"
            final_tg = eval_mat(target_model, best_seq)
            
            # Use alarm for final parsing just in case
            signal.alarm(2)
            tanimoto = np.nan
            try:
                best_mol = Chem.MolFromSmiles(best_str)
                if best_mol and src_fp:
                    cand_fp = AllChem.GetMorganFingerprintAsBitVect(best_mol, 2, nBits=2048)
                    tanimoto = DataStructs.TanimotoSimilarity(src_fp, cand_fp)
            except:
                pass
            finally:
                signal.alarm(0)
                
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
            print(f"Done source {si+1}/30. Props: {prop_attempts}, Cands: {valid_cands}")

    pd.DataFrame(results).to_csv(out_dir / f"per_example_results_untrained.csv", index=False)
    print(f"[materials] {cond_name}: wrote {len(results)} rows")

if __name__ == "__main__":
    run_materials_untrained()
