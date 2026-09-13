import json
import time
import hashlib
from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.data.tokenizer import tokenize
from materials_adv.data.scaler import TargetScaler

# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def randomize_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    except:
        pass
    return smiles

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class AblationDataset(Dataset):
    def __init__(self, data: list[dict], vocab_map: dict, max_len: int):
        self.data = data
        self.vocab_map = vocab_map
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def _encode(self, smi: str):
        toks_str = tokenize(smi)[:self.max_len]
        toks = [self.vocab_map.get(t, 0) for t in toks_str]
        mask = [False]*len(toks) + [True]*(self.max_len - len(toks))
        toks = toks + [0]*(self.max_len - len(toks))
        return torch.tensor(toks, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, i):
        rec = self.data[i]
        c_t, c_m = self._encode(rec["original_representation"])
        a_t, a_m = self._encode(rec["adversarial_representation"])
        rnd = randomize_smiles(rec["original_representation"])
        r_t, r_m = self._encode(rnd)
        
        return {
            "source_id": rec["source_id"],
            "c_t": c_t, "c_m": c_m,
            "a_t": a_t, "a_m": a_m,
            "r_t": r_t, "r_m": r_m,
            "tgt": torch.tensor(rec["property_value"], dtype=torch.float32)
        }

# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, val_df, vocab_map, scaler, max_len, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, row in val_df.iterrows():
            t, m = AblationDataset([{"original_representation": row["original_representation"], "adversarial_representation": "", "source_id": "", "property_value": 0}], vocab_map, max_len)._encode(row["original_representation"])
            pred = model(t.unsqueeze(0).to(device), padding_mask=m.unsqueeze(0).to(device)).squeeze()
            y_pred.append(pred.item())
            y_true.append(row["property_value"])
    
    y_true = np.array(y_true)
    y_pred = scaler.inverse_transform(np.array(y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), 
            "std": float(np.std(y_pred)), "min": float(np.min(y_pred)), "max": float(np.max(y_pred)), "preds": y_pred.tolist()}

# ─────────────────────────────────────────────────────────────
# Training Engine
# ─────────────────────────────────────────────────────────────
def get_param_drift(d1, d0_state):
    total = 0.0
    enc = 0.0
    head = 0.0
    for name, p in d1.named_parameters():
        p0 = d0_state[name].to(p.device)
        diff = torch.sum((p - p0)**2).item()
        total += diff
        if "encoder" in name or "embedding" in name:
            enc += diff
        else:
            head += diff
    return np.sqrt(total), np.sqrt(enc), np.sqrt(head)

def train_variant(name: str, config: dict, d0_state: dict, dataset: AblationDataset, 
                  val_df: pd.DataFrame, vocab_map: dict, scaler: TargetScaler, 
                  arch: dict, device: str):
    print(f"\n[Training Variant: {name}]")
    torch.manual_seed(20260901)
    
    model = TwoBranchTransformerRegressorModel(
        vocab_size=len(vocab_map), d_model=arch["d_model"], n_layers=arch["n_layers"],
        n_heads=arch["n_heads"], dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"], max_seq_len=arch["max_seq_len"],
        pooling=arch["pooling"], branch_dim=arch["branch_dim"],
    )
    model.load_state_dict(d0_state)
    model.to(device)
    
    if config["freeze_encoder"]:
        for n, p in model.named_parameters():
            if "encoder" in n or "embedding" in n:
                p.requires_grad_(False)
                
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    history = []
    best_mae = float("inf")
    best_state = None
    
    n_clean = int(32 * config["clean_ratio"])
    
    for epoch in range(1, 41):  # 40 epochs as in phase 9
        model.train()
        l_c_ep, l_a_ep = 0.0, 0.0
        g_c_ep, g_a_ep = 0.0, 0.0
        
        lam = config["lambda_fn"](epoch)
        
        for batch in loader:
            opt.zero_grad()
            c_t = batch["c_t"].to(device); c_m = batch["c_m"].to(device)
            a_t = batch["a_t"].to(device); a_m = batch["a_m"].to(device)
            r_t = batch["r_t"].to(device); r_m = batch["r_m"].to(device)
            tgt = batch["tgt"].to(device)
            
            # Batch splitting
            B = c_t.size(0)
            nc = int(B * config["clean_ratio"])
            
            if nc == B: # 100% clean
                pred_c = model(c_t, padding_mask=c_m).squeeze(-1)
                loss = F.mse_loss(pred_c, tgt)
                l_c_ep += loss.item()
            else:
                pred_c = model(c_t[:nc], padding_mask=c_m[:nc]).squeeze(-1)
                l_clean = F.mse_loss(pred_c, tgt[:nc])
                
                # Adversarial or Random replay
                adv_t = r_t if config["use_random"] else a_t
                adv_m = r_m if config["use_random"] else a_m
                
                pred_a = model(adv_t[nc:], padding_mask=adv_m[nc:]).squeeze(-1)
                l_adv = F.mse_loss(pred_a, tgt[nc:])
                
                loss = l_clean + lam * l_adv
                l_c_ep += l_clean.item()
                l_a_ep += l_adv.item()
                
                # Tracking gradients roughly on first few epochs if possible
                if epoch <= 2 and hasattr(opt, "zero_grad"):
                    # compute clean grad
                    l_clean.backward(retain_graph=True)
                    gn_c = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                    g_c_ep += gn_c
                    opt.zero_grad()
                    
                    # compute adv grad
                    (lam * l_adv).backward(retain_graph=True)
                    gn_a = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                    g_a_ep += gn_a
                    opt.zero_grad()
                    
            loss.backward()
            opt.step()
            
        val = evaluate_model(model, val_df, vocab_map, scaler, arch["max_seq_len"], device)
        if val["mae"] < best_mae:
            best_mae = val["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        history.append({
            "epoch": epoch, "l_clean": l_c_ep/len(loader), "l_adv": l_a_ep/len(loader),
            "val_mae": val["mae"], "val_r2": val["r2"], "lambda": lam
        })
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d} | val_mae={val['mae']:.4f} | lam={lam:.3f} | Lc={l_c_ep/len(loader):.3f} La={l_a_ep/len(loader):.3f}")
            
    model.load_state_dict(best_state)
    val_final = evaluate_model(model, val_df, vocab_map, scaler, arch["max_seq_len"], device)
    drift_tot, drift_enc, drift_head = get_param_drift(model, d0_state)
    
    return model, {
        "val_mae": val_final["mae"], "val_rmse": val_final["rmse"], "val_r2": val_final["r2"],
        "pred_std": val_final["std"], "pred_min": val_final["min"], "pred_max": val_final["max"],
        "drift_tot": float(drift_tot), "drift_enc": float(drift_enc), "drift_head": float(drift_head),
        "history": history, "clean_preds": val_final["preds"]
    }

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    root = Path(__file__).resolve().parent.parent
    p9_dir = root / "results" / "phase9_representation_closed_loop" / "1789295546"
    p4_dir = root / "results" / "phase4_controlled_robustness"
    
    run_id = str(int(time.time()))
    out_dir = root / "results" / "phase10_collapse_diagnosis" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Phase 10 | Device: {device} | Run ID: {run_id}")
    
    # ── Configs ──────────────────────────────────────────────
    import yaml
    with (root / "configs" / "specialized_transformer.yaml").open() as f:
        cfg = yaml.safe_load(f)
    arch = cfg["architecture"]
    
    # ── Load Data ────────────────────────────────────────────
    df = pd.read_csv(root / "data" / "processed" / "processed.csv")
    with (root / "data" / "processed" / "splits.json").open() as f:
        splits = json.load(f)
    with (root / "data" / "processed" / "vocab.json").open() as f:
        vocab = json.load(f)
        
    vocab_map = {tok: idx + 1 for idx, tok in enumerate(vocab)}
    val_df = df.iloc[splits["val"]].copy()
    
    # Load Phase 9 Adv Training Set
    adv_ds_path = p9_dir / "adversarial_training_set.jsonl"
    adv_hash = sha256(adv_ds_path)
    print(f"Phase 9 Adv Dataset Hash: {adv_hash}")
    
    train_data = []
    with adv_ds_path.open() as f:
        for l in f:
            rec = json.loads(l)
            # Find property value from df
            prop = df.loc[df["polymer_id"] == rec["source_id"], "property_value"].iloc[0]
            rec["property_value"] = float(prop)
            train_data.append(rec)
            
    dataset = AblationDataset(train_data, vocab_map, arch["max_seq_len"])
    
    # Load D0
    d0_ckpt = p4_dir / "mix_robust_0.1"
    d0_state = torch.load(d0_ckpt / "model.pt", map_location=device, weights_only=True)
    scaler = TargetScaler.load(root / "results" / "models" / "transformer_regressor" / "scaler.json")
    
    # Eval D0
    d0_model = TwoBranchTransformerRegressorModel(
        vocab_size=len(vocab_map), d_model=arch["d_model"], n_layers=arch["n_layers"],
        n_heads=arch["n_heads"], dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"], max_seq_len=arch["max_seq_len"],
        pooling=arch["pooling"], branch_dim=arch["branch_dim"],
    )
    d0_model.load_state_dict(d0_state)
    d0_model.to(device)
    d0_val = evaluate_model(d0_model, val_df, vocab_map, scaler, arch["max_seq_len"], device)
    
    # ── Ablation Variants ────────────────────────────────────
    variants = {
        "A_FailedOriginal": {"lr": 1e-4, "clean_ratio": 0.75, "use_random": False, "freeze_encoder": False, "lambda_fn": lambda e: 0.1},
        "B_CleanRandom":    {"lr": 1e-4, "clean_ratio": 0.75, "use_random": True,  "freeze_encoder": False, "lambda_fn": lambda e: 0.1},
        "C_CleanLowAdv":    {"lr": 1e-4, "clean_ratio": 0.75, "use_random": False, "freeze_encoder": False, "lambda_fn": lambda e: 0.01},
        "D_Curriculum":     {"lr": 1e-4, "clean_ratio": 0.75, "use_random": False, "freeze_encoder": False, "lambda_fn": lambda e: 0.0 if e<=5 else (0.01 if e<=10 else (0.05 if e<=15 else 0.1))},
        "E_FrozenEncoder":  {"lr": 1e-4, "clean_ratio": 0.75, "use_random": False, "freeze_encoder": True,  "lambda_fn": lambda e: 0.1},
    }
    
    results = {}
    models = {"D0": d0_model}
    
    for v_name, v_cfg in variants.items():
        m, stats = train_variant(v_name, v_cfg, d0_state, dataset, val_df, vocab_map, scaler, arch, device)
        models[v_name] = m
        results[v_name] = stats
        
        # Save training curves
        pd.DataFrame(stats["history"]).to_csv(out_dir / f"curves_{v_name}.csv", index=False)
        
    # ── Catastrophic Forgetting ───────────────────────────────
    forget_stats = []
    d0_preds = np.array(d0_val["preds"])
    for name, stats in results.items():
        v_preds = np.array(stats["clean_preds"])
        shift = np.abs(v_preds - d0_preds)
        forget_stats.append({
            "variant": name,
            "mean_shift": float(np.mean(shift)),
            "median_shift": float(np.median(shift)),
            "p95_shift": float(np.percentile(shift, 95))
        })
    pd.DataFrame(forget_stats).to_csv(out_dir / "clean_prediction_shift.csv", index=False)
    
    # ── Representation Robustness (Fresh vs Seen) ────────────
    print("\n[Evaluating Robustness]")
    fresh_attack_file = p9_dir / "fresh_attack_D0.jsonl"
    fresh_data = [json.loads(l) for l in fresh_attack_file.open()]
    
    # We will evaluate mean drift on seen (train) and unseen (fresh)
    seen_drifts = {k: [] for k in models.keys()}
    fresh_drifts = {k: [] for k in models.keys()}
    
    def get_preds(m, smis):
        t_batch, m_batch = [], []
        for s in smis:
            t, mask = AblationDataset([{"original_representation":s,"adversarial_representation":"","source_id":"","property_value":0}], vocab_map, arch["max_seq_len"])._encode(s)
            t_batch.append(t); m_batch.append(mask)
        with torch.no_grad():
            p = m(torch.stack(t_batch).to(device), padding_mask=torch.stack(m_batch).to(device)).squeeze(-1)
        return scaler.inverse_transform(p.cpu().numpy())
    
    # Seen evaluation (subsample 100 for speed)
    seen_subset = train_data[:100]
    for k, m in models.items():
        m.eval()
        p_orig = get_preds(m, [r["original_representation"] for r in seen_subset])
        p_adv = get_preds(m, [r["adversarial_representation"] for r in seen_subset])
        seen_drifts[k] = np.mean(np.abs(p_orig - p_adv))
        
        # Fresh evaluation using model-specific adaptive attack
        rng_fresh = np.random.default_rng(77777777)
        subset_val = val_df.sort_values("polymer_id").head(100).copy()
        
        m_fresh_drifts = []
        for _, row in subset_val.iterrows():
            res = best_of_n_repr_attack(
                m, row["original_representation"], vocab_map,
                arch["max_seq_len"], device, rng_fresh, query_budget=50
            )
            m_fresh_drifts.append(res["drift"])
        fresh_drifts[k] = np.mean(m_fresh_drifts)
        
    rob_df = pd.DataFrame([{
        "variant": k, "seen_drift": seen_drifts[k], "fresh_drift": fresh_drifts[k],
        "overfit_ratio": fresh_drifts[k] / (seen_drifts[k] + 1e-6)
    } for k in models.keys()])
    rob_df.to_csv(out_dir / "adversarial_overfitting.csv", index=False)
    print(rob_df)
    
    # ── Embedding Collapse ───────────────────────────────────
    print("\n[Evaluating Embedding Collapse]")
    emb_stats = []
    # get 100 diverse validations
    val_smis = val_df["original_representation"].head(100).tolist()
    t_batch, m_batch = [], []
    for s in val_smis:
        t, mask = AblationDataset([{"original_representation":s,"adversarial_representation":"","source_id":"","property_value":0}], vocab_map, arch["max_seq_len"])._encode(s)
        t_batch.append(t); m_batch.append(mask)
    tb = torch.stack(t_batch).to(device)
    mb = torch.stack(m_batch).to(device)
    
    for k, m in models.items():
        with torch.no_grad():
            embs = m.encode_shared(tb, padding_mask=mb)
            embs = embs.cpu().numpy()
            
            norms = np.linalg.norm(embs, axis=1)
            mean_norm = np.mean(norms)
            var_dim = np.var(embs, axis=0).mean()
            # pairwise cosine sim
            norms_expanded = norms[:, np.newaxis]
            sims = np.dot(embs, embs.T) / (norms_expanded @ norms_expanded.T + 1e-9)
            np.fill_diagonal(sims, np.nan)
            mean_sim = np.nanmean(sims)
            
            emb_stats.append({"variant": k, "mean_norm": float(mean_norm), "mean_variance": float(var_dim), "mean_pairwise_sim": float(mean_sim)})
            
    pd.DataFrame(emb_stats).to_csv(out_dir / "embedding_collapse.csv", index=False)
    
    # ── Metrics aggregation ──────────────────────────────────
    metrics_all = []
    for k in variants.keys():
        r = results[k]
        metrics_all.append({
            "variant": k, "val_mae": r["val_mae"], "val_r2": r["val_r2"],
            "pred_std": r["pred_std"], "drift_tot": r["drift_tot"], "drift_enc": r["drift_enc"], "drift_head": r["drift_head"]
        })
    pd.DataFrame(metrics_all).to_csv(out_dir / "model_metrics.csv", index=False)
    
    # ── Hypothesis Verdicts ──────────────────────────────────
    # Check if any variant succeeds: clean degradation <= 0.02 and fresh drift < D0 fresh drift by meaningful amount
    d0_mae = d0_val["mae"]
    d0_fresh = fresh_drifts["D0"]
    
    success_variant = None
    for k in variants.keys():
        if results[k]["val_mae"] - d0_mae <= 0.02 and fresh_drifts[k] < d0_fresh - 0.05:
            success_variant = k
            break
            
    verdicts = {
        "H1_excessive_weight": "SUPPORTED" if results["C_CleanLowAdv"]["val_r2"] > 0.70 else "NOT SUPPORTED",
        "H2_catastrophic_forgetting": "SUPPORTED" if np.mean(np.abs(results["C_CleanLowAdv"]["clean_preds"] - d0_preds)) > 0.15 else "NOT SUPPORTED",
        "H3_worst_case_bias": "SUPPORTED" if results["B_CleanRandom"]["val_r2"] > results["A_FailedOriginal"]["val_r2"] else "NOT SUPPORTED",
        "H4_frozen_encoder_preserves": "SUPPORTED" if results["E_FrozenEncoder"]["val_r2"] > 0.70 else "NOT SUPPORTED",
        "H5_curriculum_avoids": "SUPPORTED" if results["D_Curriculum"]["val_r2"] > results["A_FailedOriginal"]["val_r2"] else "NOT SUPPORTED",
        "H6_random_stable": "SUPPORTED" if results["B_CleanRandom"]["val_r2"] > 0.70 else "NOT SUPPORTED",
        "success_variant": success_variant,
        "d0_hash": d0_hash,
        "adv_dataset_hash": adv_hash
    }
    with (out_dir / "hypothesis_verdicts.json").open("w") as f:
        json.dump(verdicts, f, indent=2)
        
    print(f"\nPhase 10 Output written to {out_dir}")
import torch
import numpy as np
from rdkit import Chem

def canonical(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def best_of_n_repr_attack(
    model,
    orig_rep: str,
    vocab_map: dict,
    max_seq_len: int,
    device: str,
    rng: np.random.Generator,
    query_budget: int = 50,
    batch_size: int = 25,
) -> dict:
    canon_orig = canonical(orig_rep)
    if canon_orig is None:
        return {"found": False, "adversarial_representation": orig_rep,
                "drift": 0.0, "query_count": 0, "valid_rate": 0.0}

    mol = Chem.MolFromSmiles(orig_rep)
    if mol is None:
        return {"found": False, "adversarial_representation": orig_rep,
                "drift": 0.0, "query_count": 0, "valid_rate": 0.0}

    # Original prediction
    dummy_ds = AblationDataset([], vocab_map, max_seq_len)
    orig_t, orig_m = dummy_ds._encode(orig_rep)
    with torch.no_grad():
        orig_pred = model(orig_t.unsqueeze(0).to(device),
                          padding_mask=orig_m.unsqueeze(0).to(device)).item()

    seen = {orig_rep}
    queried: list[dict] = []
    best_drift = 0.0
    best_cand  = orig_rep
    q_count    = 0

    while q_count < query_budget:
        remaining = query_budget - q_count
        n_gen = min(batch_size * 4, remaining * 4)
        rdkit_seed = int(rng.integers(1, 2**31 - 1))
        candidates_raw = Chem.MolToRandomSmilesVect(mol, n_gen, randomSeed=rdkit_seed)

        # Filter to canonical-equivalent, unseen candidates
        batch_cands = []
        for cand in candidates_raw:
            if q_count + len(batch_cands) >= query_budget:
                break
            if cand in seen:
                continue
            seen.add(cand)
            if canonical(cand) != canon_orig:
                continue
            batch_cands.append(cand)

        if not batch_cands:
            # Generate a larger burst to escape duplicates
            rdkit_seed2 = int(rng.integers(1, 2**31 - 1))
            extra = Chem.MolToRandomSmilesVect(mol, n_gen * 2, randomSeed=rdkit_seed2)
            for cand in extra:
                if q_count + len(batch_cands) >= query_budget:
                    break
                if cand in seen:
                    continue
                seen.add(cand)
                if canonical(cand) == canon_orig:
                    batch_cands.append(cand)
            if not batch_cands:
                break  # exhausted distinct equivalents

        # Batch encode
        ts = torch.stack([dummy_ds._encode(c)[0] for c in batch_cands]).to(device)
        ms = torch.stack([dummy_ds._encode(c)[1] for c in batch_cands]).to(device)

        with torch.no_grad():
            preds = model(ts, padding_mask=ms).squeeze(-1).cpu().numpy()
        preds = np.atleast_1d(preds)  # guard against 0-d array when batch_size==1

        for cand, pred in zip(batch_cands, preds):
            q_count += 1
            drift = float(abs(pred - orig_pred))
            queried.append({"rep": cand, "pred": float(pred), "drift": drift})
            if drift > best_drift:
                best_drift = drift
                best_cand  = cand

    valid_rate = len(queried) / max(q_count, 1)
    return {
        "found": best_cand != orig_rep,
        "adversarial_representation": best_cand,
        "canonical_original": canon_orig,
        "canonical_adversarial": canonical(best_cand),
        "drift": best_drift,
        "original_prediction": orig_pred,
        "adversarial_prediction": next(
            (q["pred"] for q in queried if q["rep"] == best_cand), orig_pred
        ),
        "query_count": q_count,
        "valid_rate": valid_rate,
        "queried": queried,
    }
if __name__ == "__main__":
    main()
