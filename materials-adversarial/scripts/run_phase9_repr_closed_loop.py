#!/usr/bin/env python3
"""Phase 9 (Corrected): Representation-Preserving Closed-Loop Adversarial Training.

Uses ONLY canonically equivalent polymer string representations as adversarial
training examples. Molecule identity is preserved, so measured Bandgap labels
are scientifically valid for adversarial supervision.

Pipeline:
  1. Load and freeze D0 (mixed-robust Phase 4 model).
  2. Run best-of-N adaptive search over equivalent representations on TRAIN split.
  3. Train D1 (initialized from D0) with L_clean + lambda_adv * L_adv + lambda_cons * L_cons.
  4. Select best lambda_adv on validation only (clean MAE degradation <= 0.02 eV).
  5. Freeze D1.
  6. Fresh re-attack D0 and D1 with new seeds on validation sources.
  7. Compare paired closed-loop improvement, check overfitting, check frozen banks.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from materials_adv.domain.chemistry.attacks.randomization import SmilesRandomizationAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel

# ─────────────────────────────────────────────────────────────
# Constants and config
# ─────────────────────────────────────────────────────────────
TRAIN_ATTACK_SEED = 20260901   # deterministic for training adversarial generation
FRESH_ATTACK_SEED = 77777777   # deliberately different from training seed
N_TRAIN_SOURCES  = 500         # deterministic subset of TRAIN split
QUERY_BUDGET     = 50          # Q per source
STRESS_THRESHOLD = 0.10        # eV; "stress success" if drift >= this
LAMBDA_ADV_GRID  = [0.1, 0.5]
LAMBDA_CONS      = 1.0         # kept from D0
CLEAN_MAE_TOL    = 0.02        # max allowed validation MAE degradation vs D0


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    if path.exists():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def vocab_hash(vocab: list[str]) -> str:
    return hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()


def encode(rep: str, vocab_map: dict, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenize(rep)
    seq_len = min(len(tokens), max_seq_len)
    t = torch.zeros(max_seq_len, dtype=torch.long)
    m = torch.ones(max_seq_len, dtype=torch.bool)
    for j in range(seq_len):
        t[j] = vocab_map.get(tokens[j], 0)
        m[j] = False
    return t, m


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_model(vocab: list[str], arch: dict, ckpt_dir: Path, device: str) -> TwoBranchTransformerRegressorModel:
    m = TwoBranchTransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        pooling=arch["pooling"],
        branch_dim=arch.get("branch_dim", 32),
    )
    m.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True))
    m.to(device)
    m.eval()
    return m


def predict_batch(model, reps: list[str], vocab_map: dict, max_seq_len: int, device: str) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for rep in reps:
            t, m = encode(rep, vocab_map, max_seq_len)
            t, m = t.unsqueeze(0).to(device), m.unsqueeze(0).to(device)
            preds.append(model(t, padding_mask=m).item())
    return np.array(preds)


def evaluate_model(model, df: pd.DataFrame, vocab_map: dict, scaler: TargetScaler,
                   arch: dict, device: str) -> dict:
    reps = df["original_representation"].tolist()
    targets = df["property_value"].values
    preds_norm = predict_batch(model, reps, vocab_map, arch["max_seq_len"], device)
    preds = scaler.inverse_transform(preds_norm)
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_r = np.sum((targets - preds) ** 2)
    ss_t = np.sum((targets - np.mean(targets)) ** 2)
    r2   = float(1.0 - ss_r / ss_t)
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(reps),
            "pred_std": float(np.std(preds)), "pred_min": float(np.min(preds)),
            "pred_max": float(np.max(preds))}


# ─────────────────────────────────────────────────────────────
# Representation-Preserving Adaptive Search
# ─────────────────────────────────────────────────────────────
def best_of_n_repr_attack(
    model,
    orig_rep: str,
    vocab_map: dict,
    max_seq_len: int,
    device: str,
    rng: np.random.Generator,
    query_budget: int = QUERY_BUDGET,
    batch_size: int = 25,
) -> dict:
    """Best-of-N adaptive search over canonically equivalent representations.

    Generates batches of random SMILES for the same molecule, scores each
    against the model in a single batched forward pass, and retains the
    highest-drift equivalent string found.

    All candidates must satisfy:
        canonical(candidate) == canonical(original)
        candidate != original
    """
    canon_orig = canonical(orig_rep)
    if canon_orig is None:
        return {"found": False, "adversarial_representation": orig_rep,
                "drift": 0.0, "query_count": 0, "valid_rate": 0.0}

    mol = Chem.MolFromSmiles(orig_rep)
    if mol is None:
        return {"found": False, "adversarial_representation": orig_rep,
                "drift": 0.0, "query_count": 0, "valid_rate": 0.0}

    # Original prediction
    orig_t, orig_m = encode(orig_rep, vocab_map, max_seq_len)
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
        ts = torch.stack([encode(c, vocab_map, max_seq_len)[0] for c in batch_cands]).to(device)
        ms = torch.stack([encode(c, vocab_map, max_seq_len)[1] for c in batch_cands]).to(device)

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


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class ReprAdvDataset(Dataset):
    def __init__(self, clean_reps, rand_reps, adv_reps, targets_norm, vocab_map, max_seq_len):
        self.clean = clean_reps
        self.rand  = rand_reps
        self.adv   = adv_reps
        self.tgts  = targets_norm
        self.vocab_map = vocab_map
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        c_t, c_m = encode(self.clean[idx], self.vocab_map, self.max_seq_len)
        r_t, r_m = encode(self.rand[idx],  self.vocab_map, self.max_seq_len)
        a_t, a_m = encode(self.adv[idx],   self.vocab_map, self.max_seq_len)
        return {
            "c_t": c_t, "c_m": c_m,
            "r_t": r_t, "r_m": r_m,
            "a_t": a_t, "a_m": a_m,
            "tgt": torch.tensor(self.tgts[idx], dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
def train_d1(
    d0_model, dataset: ReprAdvDataset, tr_cfg: dict, arch: dict,
    vocab: list[str], device: str, lambda_adv: float,
    val_df: pd.DataFrame, vocab_map: dict, scaler: TargetScaler,
    seed: int = 20260901,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize from D0
    model = load_model.__wrapped__ if hasattr(load_model, "__wrapped__") else None
    model = TwoBranchTransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        pooling=arch["pooling"],
        branch_dim=arch["branch_dim"],
    )
    model.load_state_dict(d0_model.state_dict())  # warm-start from D0
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["learning_rate"],
                             weight_decay=tr_cfg["weight_decay"])
    loader = DataLoader(dataset, batch_size=tr_cfg["batch_size"], shuffle=True)

    best_mae   = float("inf")
    best_state = None
    history    = []

    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            c_t = batch["c_t"].to(device); c_m = batch["c_m"].to(device)
            r_t = batch["r_t"].to(device); r_m = batch["r_m"].to(device)
            a_t = batch["a_t"].to(device); a_m = batch["a_m"].to(device)
            tgt = batch["tgt"].to(device)

            pred_c = model(c_t, padding_mask=c_m).squeeze(-1)
            pred_r = model(r_t, padding_mask=r_m).squeeze(-1)
            pred_a = model(a_t, padding_mask=a_m).squeeze(-1)

            l_clean = F.mse_loss(pred_c, tgt)
            l_cons  = F.mse_loss(pred_c, pred_r)   # existing randomization consistency
            l_adv   = F.mse_loss(pred_a, tgt)       # adversarial repr uses same measured label
            l_cons_adv = F.mse_loss(pred_a, pred_c) # representation consistency

            loss = l_clean + LAMBDA_CONS * l_cons + lambda_adv * l_adv + lambda_adv * l_cons_adv
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        val = evaluate_model(model, val_df, vocab_map, scaler, arch, device)
        if val["mae"] < best_mae:
            best_mae   = val["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "loss": ep_loss / len(loader), "val_mae": val["mae"]})
        if epoch % 5 == 0:
            print(f"    Epoch {epoch:3d} | loss={ep_loss/len(loader):.4f} | val_mae={val['mae']:.4f}")

    model.load_state_dict(best_state)
    return model, history, best_mae, best_state


# ─────────────────────────────────────────────────────────────
# Attack statistics helper
# ─────────────────────────────────────────────────────────────
def attack_stats(drifts: list[float]) -> dict:
    a = np.array(drifts)
    n = len(a)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0,
                "p95": 0.0, "max": 0.0, "stress_rate": 0.0}
    return {
        "n": n,
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "stress_rate": float(np.mean(a >= STRESS_THRESHOLD)),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    run_id  = str(int(time.time()))
    root    = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed"
    p4_dir  = root / "results" / "phase4_controlled_robustness"
    out_dir = root / "results" / "phase9_representation_closed_loop" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    d1_dir  = out_dir / "D1_checkpoint"
    d1_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  run_id: {run_id}")

    # ── Load config ──────────────────────────────────────────
    import yaml
    with (root / "configs" / "specialized_transformer.yaml").open() as f:
        cfg = yaml.safe_load(f)
    arch   = cfg["architecture"]
    tr_cfg = cfg["training"]

    # ── Load data ────────────────────────────────────────────
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open() as f:
        splits = json.load(f)
    with (data_dir / "vocab.json").open() as f:
        vocab = json.load(f)

    vocab_map = {tok: idx + 1 for idx, tok in enumerate(vocab)}
    train_df  = df.iloc[splits["train"]]
    val_df    = df.iloc[splits["val"]]

    # ── D0: Load and freeze ───────────────────────────────────
    print("\n[1] Loading D0 (mixed-robust)...")
    d0_ckpt  = p4_dir / "mix_robust_0.1"
    d0_hash  = sha256(d0_ckpt / "model.pt")

    scaler_path = root / "results" / "models" / "transformer_regressor" / "scaler.json"
    scaler = TargetScaler.load(scaler_path)
    scaler_hash_val = sha256(scaler_path)

    d0_model = load_model(vocab, arch, d0_ckpt, device)
    for p in d0_model.parameters():
        p.requires_grad_(False)

    d0_val = evaluate_model(d0_model, val_df, vocab_map, scaler, arch, device)
    print(f"  D0 val MAE={d0_val['mae']:.4f}  RMSE={d0_val['rmse']:.4f}  R²={d0_val['r2']:.4f}")

    # ── Step 2: Generate adversarial training set ─────────────
    print(f"\n[2] Adaptive representation search on {N_TRAIN_SOURCES} TRAIN sources (Q={QUERY_BUDGET})...")
    rng_train = np.random.default_rng(TRAIN_ATTACK_SEED)

    # Deterministic subset: first N_TRAIN_SOURCES by sorted polymer_id
    subset = train_df.sort_values("polymer_id").head(N_TRAIN_SOURCES).copy()
    source_manifest = [
        {"source_id": row["polymer_id"], "original_representation": row["original_representation"],
         "property_value": float(row["property_value"])}
        for _, row in subset.iterrows()
    ]
    with (out_dir / "training_source_manifest.json").open("w") as f:
        json.dump(source_manifest, f, indent=2)

    attack_records = []
    for i, (_, row) in enumerate(subset.iterrows()):
        res = best_of_n_repr_attack(
            d0_model, row["original_representation"], vocab_map,
            arch["max_seq_len"], device, rng_train, query_budget=QUERY_BUDGET,
        )
        rec = {
            "source_id": row["polymer_id"],
            "original_representation": row["original_representation"],
            "adversarial_representation": res["adversarial_representation"],
            "canonical_original": res.get("canonical_original"),
            "canonical_adversarial": res.get("canonical_adversarial"),
            "D0_original_prediction": res.get("original_prediction"),
            "D0_adversarial_prediction": res.get("adversarial_prediction"),
            "drift_D0": float(res["drift"]),
            "query_count": int(res["query_count"]),
            "valid_rate": float(res["valid_rate"]),
            "attack_seed": TRAIN_ATTACK_SEED,
            "found": bool(res["found"]),
            "label_basis": "measured_physical_supervision",  # molecule unchanged
        }
        attack_records.append(rec)
        if (i + 1) % 50 == 0:
            found = sum(r["found"] for r in attack_records)
            print(f"  [{i+1}/{N_TRAIN_SOURCES}]  found={found}  mean_drift={np.mean([r['drift_D0'] for r in attack_records]):.4f}")

    with (out_dir / "representation_attack_candidates.jsonl").open("w") as f:
        for rec in attack_records:
            # don't store full queried list in the per-record file (too large)
            f.write(json.dumps({k: v for k, v in rec.items() if k != "queried"}) + "\n")

    # Only keep sources where we found a distinct adversarial representation
    valid_records = [r for r in attack_records if r["found"]]
    print(f"  Valid adversarial representations: {len(valid_records)}/{N_TRAIN_SOURCES} "
          f"({100*len(valid_records)/N_TRAIN_SOURCES:.1f}%)")

    with (out_dir / "adversarial_training_set.jsonl").open("w") as f:
        for rec in valid_records:
            f.write(json.dumps(rec) + "\n")

    # Attack statistics on training sources
    train_drifts = [r["drift_D0"] for r in attack_records]
    print(f"  Training attack: mean_drift={np.mean(train_drifts):.4f}  "
          f"p90={np.percentile(train_drifts, 90):.4f}  max={max(train_drifts):.4f}")

    # ── Step 3: Build D1 training dataset ────────────────────
    print("\n[3] Building D1 training dataset...")
    adv_lookup = {r["source_id"]: r["adversarial_representation"] for r in valid_records}
    valid_subset = subset[subset["polymer_id"].isin(adv_lookup)]

    clean_reps = valid_subset["original_representation"].tolist()
    targets_raw = valid_subset["property_value"].values
    targets_norm = scaler.transform(targets_raw).tolist()

    # Generate randomized SMILES pairs (same as D0 training consistency)
    rng_rand = np.random.default_rng(TRAIN_ATTACK_SEED + 1)
    rand_attack = SmilesRandomizationAttack(rng_rand, n_attempts=10)

    rand_reps = []
    for rep in clean_reps:
        tokens = tokenize(rep)
        outcomes = rand_attack.generate(tokens, n_variants=1)
        if outcomes:
            rand_reps.append(outcomes[0].adversarial_representation)
        else:
            rand_reps.append(rep)

    adv_reps = [adv_lookup[pid] for pid in valid_subset["polymer_id"]]

    dataset = ReprAdvDataset(clean_reps, rand_reps, adv_reps, targets_norm, vocab_map, arch["max_seq_len"])
    print(f"  Training set size: {len(dataset)} samples")

    # ── Step 4: Train D1 for each lambda_adv ─────────────────
    print("\n[4] Training D1 variants...")
    lambda_rows = []
    best_lambda = None
    best_d1_model = None
    best_d1_state = None
    best_d1_hist  = None
    best_d1_mae   = float("inf")

    for l_adv in LAMBDA_ADV_GRID:
        print(f"\n  lambda_adv={l_adv}")
        model, hist, val_mae, state = train_d1(
            d0_model, dataset, tr_cfg, arch, vocab, device,
            l_adv, val_df, vocab_map, scaler,
        )
        mae_deg = val_mae - d0_val["mae"]
        feasible = mae_deg <= CLEAN_MAE_TOL
        row = {"lambda_adv": l_adv, "val_mae": val_mae, "mae_degradation": mae_deg, "feasible": feasible}
        lambda_rows.append(row)
        print(f"    val_mae={val_mae:.4f}  degradation={mae_deg:+.4f}  feasible={feasible}")

        if feasible and (best_lambda is None or l_adv > best_lambda):
            best_lambda   = l_adv
            best_d1_model = model
            best_d1_state = state
            best_d1_hist  = hist
            best_d1_mae   = val_mae

    if best_lambda is None:
        print("  Warning: all lambda_adv exceeded MAE tolerance; selecting lambda_adv=0.1 as fallback.")
        best_lambda   = LAMBDA_ADV_GRID[0]
        best_d1_model = model      # last trained
        best_d1_state = state
        best_d1_hist  = hist
        best_d1_mae   = val_mae

    print(f"\n  Selected lambda_adv = {best_lambda}")
    pd.DataFrame(lambda_rows).to_csv(out_dir / "lambda_selection.csv", index=False)
    pd.DataFrame(best_d1_hist).to_csv(out_dir / "training_history.csv", index=False)

    # Freeze D1
    for p in best_d1_model.parameters():
        p.requires_grad_(False)

    torch.save(best_d1_state, d1_dir / "model.pt")
    scaler.save(d1_dir / "scaler.json")
    d1_hash = sha256(d1_dir / "model.pt")

    d1_val = evaluate_model(best_d1_model, val_df, vocab_map, scaler, arch, device)
    print(f"  D1 val MAE={d1_val['mae']:.4f}  RMSE={d1_val['rmse']:.4f}  R²={d1_val['r2']:.4f}")

    pd.DataFrame([{"model": "D0", **d0_val}, {"model": "D1", **d1_val}]).to_csv(
        out_dir / "clean_metrics.csv", index=False
    )

    # Save training and model config
    with (out_dir / "training_config.json").open("w") as f:
        json.dump({"lambda_adv": best_lambda, "lambda_cons": LAMBDA_CONS,
                   "n_train_sources": N_TRAIN_SOURCES, "query_budget": QUERY_BUDGET,
                   "train_attack_seed": TRAIN_ATTACK_SEED, "fresh_attack_seed": FRESH_ATTACK_SEED,
                   "clean_mae_tolerance": CLEAN_MAE_TOL, "stress_threshold": STRESS_THRESHOLD}, f, indent=2)

    # ── Step 5: Fresh re-attack D0 and D1 ────────────────────
    print("\n[5] Fresh representation re-attack on validation sources (new seeds)...")
    rng_fresh = np.random.default_rng(FRESH_ATTACK_SEED)
    assert FRESH_ATTACK_SEED != TRAIN_ATTACK_SEED

    val_sources = []
    for _, row in val_df.iterrows():
        val_sources.append({"id": row["polymer_id"], "rep": row["original_representation"],
                            "target": float(row["property_value"])})
        if len(val_sources) >= 100:
            break

    def run_fresh_attack(model, label, seeds_offset=0):
        rng_f = np.random.default_rng(FRESH_ATTACK_SEED + seeds_offset)
        records = []
        for i, src in enumerate(val_sources):
            res = best_of_n_repr_attack(
                model, src["rep"], vocab_map, arch["max_seq_len"], device,
                rng_f, query_budget=QUERY_BUDGET,
            )
            # Build query curve
            q_curve = {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
            if res.get("queried"):
                running_best = 0.0
                for q_idx, entry in enumerate(res["queried"], 1):
                    running_best = max(running_best, entry["drift"])
                    for budget in [b for b in q_curve if q_idx <= b]:
                        q_curve[budget] = max(q_curve[budget], running_best)

            records.append({
                "source_id": src["id"],
                "model": label,
                "best_drift": float(res["drift"]),
                "found": bool(res["found"]),
                "query_count": int(res["query_count"]),
                "valid_rate": float(res["valid_rate"]),
                "drift_at_Q5":  float(q_curve[5]),
                "drift_at_Q10": float(q_curve[10]),
                "drift_at_Q20": float(q_curve[20]),
                "drift_at_Q50": float(q_curve[50]),
            })
            if (i + 1) % 20 == 0:
                print(f"    {label}: [{i+1}/100]  mean_drift={np.mean([r['best_drift'] for r in records]):.4f}")
        return records

    d0_fresh = run_fresh_attack(d0_model, "D0", seeds_offset=0)
    d1_fresh = run_fresh_attack(best_d1_model, "D1", seeds_offset=1)

    with (out_dir / "fresh_attack_D0.jsonl").open("w") as f:
        for r in d0_fresh:
            f.write(json.dumps(r) + "\n")
    with (out_dir / "fresh_attack_D1.jsonl").open("w") as f:
        for r in d1_fresh:
            f.write(json.dumps(r) + "\n")

    # Query curves CSV
    curve_rows = []
    for r in d0_fresh + d1_fresh:
        curve_rows.append(r)
    pd.DataFrame(curve_rows).to_csv(out_dir / "query_curves.csv", index=False)

    # ── Step 6: Closed-loop comparison ───────────────────────
    print("\n[6] Computing closed-loop improvement...")
    d0_drifts = np.array([r["best_drift"] for r in d0_fresh])
    d1_drifts = np.array([r["best_drift"] for r in d1_fresh])
    improvements = d0_drifts - d1_drifts   # positive = D1 harder to attack

    ci_lo, ci_hi = bootstrap_ci(improvements)
    comp_rows = [
        {"source_id": d0_fresh[i]["source_id"],
         "D0_drift": d0_drifts[i], "D1_drift": d1_drifts[i],
         "improvement": improvements[i]}
        for i in range(len(d0_fresh))
    ]
    pd.DataFrame(comp_rows).to_csv(out_dir / "closed_loop_comparison.csv", index=False)

    d0_stats = attack_stats(d0_drifts.tolist())
    d1_stats = attack_stats(d1_drifts.tolist())
    print(f"  D0: mean_drift={d0_stats['mean']:.4f}  p90={d0_stats['p90']:.4f}  stress={d0_stats['stress_rate']:.2%}")
    print(f"  D1: mean_drift={d1_stats['mean']:.4f}  p90={d1_stats['p90']:.4f}  stress={d1_stats['stress_rate']:.2%}")
    print(f"  Mean improvement: {np.mean(improvements):.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Fraction improved: {np.mean(improvements > 0):.2%}  worsened: {np.mean(improvements < 0):.2%}")

    # ── Step 7: Adversarial overfitting check ─────────────────
    print("\n[7] Adversarial overfitting check...")
    # Evaluate D1 on TRAINING adversarial reps it was trained on
    if valid_records:
        d1_train_adv_preds_norm = predict_batch(
            best_d1_model, [r["adversarial_representation"] for r in valid_records],
            vocab_map, arch["max_seq_len"], device
        )
        d1_train_orig_preds_norm = predict_batch(
            best_d1_model, [r["original_representation"] for r in valid_records],
            vocab_map, arch["max_seq_len"], device
        )
        train_adv_drifts = np.abs(d1_train_adv_preds_norm - d1_train_orig_preds_norm)
    else:
        train_adv_drifts = np.array([0.0])

    overfitting_doc = [{
        "drift_D1_on_training_adversaries_mean": float(np.mean(train_adv_drifts)),
        "drift_D1_on_fresh_adversaries_mean": float(d1_stats["mean"]),
        "overfitting_gap": float(np.mean(train_adv_drifts) - d1_stats["mean"]),
        "likely_overfitting": bool(np.mean(train_adv_drifts) < 0.5 * d1_stats["mean"]),
    }]
    pd.DataFrame(overfitting_doc).to_csv(out_dir / "overfitting_check.csv", index=False)
    print(f"  D1 drift on training adversaries: {np.mean(train_adv_drifts):.4f}")
    print(f"  D1 drift on fresh adversaries:    {d1_stats['mean']:.4f}")

    # ── Step 8: Frozen bank regression ───────────────────────
    print("\n[8] Frozen bank regression check...")
    bank_dir = root / "results" / "candidate_banks" / "two_branch_validation_phase3"
    rand_bank_path = bank_dir / "randomization_candidates.jsonl"
    sub_bank_path  = bank_dir / "substitution_candidates.jsonl"
    del_bank_path  = root / "results" / "candidate_banks" / "deletion_transfer_phase5"

    bank_rows = []
    for label, model in [("D0", d0_model), ("D1", best_d1_model)]:
        row = {"model": label}
        # Randomization bank
        if rand_bank_path.exists():
            rand_recs = [json.loads(l) for l in rand_bank_path.open()][:200]
            orig_p = predict_batch(model, [r["original_representation"] for r in rand_recs], vocab_map, arch["max_seq_len"], device)
            adv_p  = predict_batch(model, [r["candidate_representation"] for r in rand_recs], vocab_map, arch["max_seq_len"], device)
            row["rand_mean_drift"] = float(np.mean(np.abs(orig_p - adv_p)))
        else:
            row["rand_mean_drift"] = None

        # Substitution bank
        if sub_bank_path.exists():
            sub_recs = [json.loads(l) for l in sub_bank_path.open()][:200]
            orig_p = predict_batch(model, [r["original_representation"] for r in sub_recs], vocab_map, arch["max_seq_len"], device)
            adv_p  = predict_batch(model, [r["candidate_representation"] for r in sub_recs], vocab_map, arch["max_seq_len"], device)
            row["sub_mean_drift"] = float(np.mean(np.abs(orig_p - adv_p)))
        else:
            row["sub_mean_drift"] = None

        # Deletion bank — uses 'candidate_representation' field name
        del_files = list(del_bank_path.glob("*.jsonl")) if del_bank_path.exists() else []
        if del_files:
            del_recs = [json.loads(l) for l in del_files[0].open()]
            # Accept either field name
            del_recs_valid = [
                r for r in del_recs
                if "original_representation" in r and
                   ("candidate_representation" in r or "adversarial_representation" in r)
            ][:200]
            if del_recs_valid:
                orig_p = predict_batch(model,
                    [r["original_representation"] for r in del_recs_valid],
                    vocab_map, arch["max_seq_len"], device)
                adv_key = ("adversarial_representation"
                           if "adversarial_representation" in del_recs_valid[0]
                           else "candidate_representation")
                adv_p = predict_batch(model,
                    [r[adv_key] for r in del_recs_valid],
                    vocab_map, arch["max_seq_len"], device)
                row["del_mean_drift"] = float(np.mean(np.abs(orig_p - adv_p)))
            else:
                row["del_mean_drift"] = None
        else:
            row["del_mean_drift"] = None

        bank_rows.append(row)
        print(f"  {label}: rand={row.get('rand_mean_drift', 'N/A'):.4f}  "
              f"sub={row.get('sub_mean_drift', 'N/A'):.4f}  "
              f"del={row.get('del_mean_drift', 'N/A') if row.get('del_mean_drift') is not None else 'N/A'}")

    pd.DataFrame(bank_rows).to_csv(out_dir / "frozen_bank_regression.csv", index=False)

    # ── Step 9: Phase 8 chemistry-changing stress regression ──
    print("\n[9] Chemistry-changing stress regression (secondary check)...")
    p8_dir = root / "results" / "phase8_multi_edit_search"
    p8_runs = sorted(p8_dir.iterdir()) if p8_dir.exists() else []
    stress_rows = []
    if p8_runs:
        traj_path = p8_runs[-1] / "trajectories.jsonl"
        if traj_path.exists():
            p8_recs = [json.loads(l) for l in traj_path.open()]
            p8_recs_metro = [r for r in p8_recs if r.get("strategy") == "metropolis"][:100]
            for label, model in [("D0", d0_model), ("D1", best_d1_model)]:
                if p8_recs_metro:
                    orig_p = predict_batch(model, [r["source_representation"] for r in p8_recs_metro], vocab_map, arch["max_seq_len"], device)
                    adv_p  = predict_batch(model, [r.get("best_candidate", r["source_representation"]) for r in p8_recs_metro], vocab_map, arch["max_seq_len"], device)
                    stress_rows.append({"model": label, "chem_stress_mean_drift": float(np.mean(np.abs(orig_p - adv_p)))})
    if not stress_rows:
        stress_rows = [{"model": "D0", "chem_stress_mean_drift": None},
                       {"model": "D1", "chem_stress_mean_drift": None}]
    pd.DataFrame(stress_rows).to_csv(out_dir / "chemistry_stress_regression.csv", index=False)

    # ── Step 10: Verdict ──────────────────────────────────────
    print("\n[10] Determining closed-loop verdict...")
    mean_imp   = float(np.mean(improvements))
    collapse_d1 = d1_val["pred_std"] < 0.5 * d0_val["pred_std"]  # prediction collapse heuristic
    mae_ok     = (d1_val["mae"] - d0_val["mae"]) <= CLEAN_MAE_TOL

    if (mean_imp > 0 and ci_lo > 0 and mae_ok and not collapse_d1):
        verdict = "SUPPORTED"
    elif (mean_imp > 0 and mae_ok and not collapse_d1):
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"

    print(f"  VERDICT: {verdict}")

    # ── Step 11: Summary ──────────────────────────────────────
    summary = {
        "label": "CANONICAL",
        "run_id": run_id,
        "verdict": verdict,
        "D0": {
            "checkpoint": str(d0_ckpt),
            "model_hash": d0_hash,
            "scaler_hash": scaler_hash_val,
            "vocab_hash": vocab_hash(vocab),
            "val_mae": d0_val["mae"], "val_rmse": d0_val["rmse"], "val_r2": d0_val["r2"],
        },
        "D1": {
            "model_hash": d1_hash,
            "selected_lambda_adv": best_lambda,
            "val_mae": d1_val["mae"], "val_rmse": d1_val["rmse"], "val_r2": d1_val["r2"],
            "mae_degradation": d1_val["mae"] - d0_val["mae"],
        },
        "training_sources": N_TRAIN_SOURCES,
        "adversarial_candidates_found": len(valid_records),
        "validity_canonical_equiv_rate": float(len(valid_records) / N_TRAIN_SOURCES),
        "D0_training_attack_stats": attack_stats(train_drifts),
        "fresh_D0_stats": d0_stats,
        "fresh_D1_stats": d1_stats,
        "paired_improvement": {
            "mean": mean_imp,
            "median": float(np.median(improvements)),
            "ci_95": [ci_lo, ci_hi],
            "fraction_improved": float(np.mean(improvements > 0)),
            "fraction_worsened": float(np.mean(improvements < 0)),
            "n": len(improvements),
        },
        "output_collapse_D1": collapse_d1,
        "overfitting": overfitting_doc[0],
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "reproducibility.json").open("w") as f:
        json.dump({
            "run_id": run_id,
            "train_attack_seed": TRAIN_ATTACK_SEED,
            "fresh_attack_seed": FRESH_ATTACK_SEED,
            "n_train_sources": N_TRAIN_SOURCES,
            "query_budget": QUERY_BUDGET,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    with (out_dir / "attack_config.json").open("w") as f:
        json.dump({"attack_type": "representation_preserving_best_of_n",
                   "canonical_equivalence_enforced": True,
                   "query_budget": QUERY_BUDGET, "search_seed": TRAIN_ATTACK_SEED}, f, indent=2)

    print(f"\nPhase 9 complete. All outputs in: {out_dir}")
    print(f"VERDICT: {verdict}")
    return summary


if __name__ == "__main__":
    main()
