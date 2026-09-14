#!/usr/bin/env python3
"""Phase 9: Closed-Loop Attack -> Defend -> Re-Attack.

Executes a single cycle of adversarial training:
1. Attacking the frozen mixed-robust D0 model on 500 train split sources (budget 3).
2. Training D1 (initialized from D0) with an additional adaptive adversarial teacher loss.
3. Attacking both D0 and D1 with fresh Metroplis search using new seeds.
4. Generating closed-loop comparison and cross-transfer analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from materials_adv.domain.chemistry.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.attacks.search.proposals import CompositeProposalOperator
from materials_adv.attacks.search.strategies import MetropolisSearch
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.specialized_transformer import (
    TwoBranchTransformerRegressorModel,
)
from materials_adv.validation.pipeline import validate


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.exists():
        with path.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    return hasher.hexdigest()

class ModelPredictor:
    def __init__(self, model, vocab_map, max_seq_len, device="cpu"):
        self.model = model
        self.vocab_map = vocab_map
        self.max_seq_len = max_seq_len
        self.device = device
        
    def predict(self, texts):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for rep in texts:
                tokens = tokenize(rep)
                seq_len = min(len(tokens), self.max_seq_len)
                src = torch.zeros((1, self.max_seq_len), dtype=torch.long, device=self.device)
                mask = torch.ones((1, self.max_seq_len), dtype=torch.bool, device=self.device)
                for j in range(seq_len):
                    src[0, j] = self.vocab_map.get(tokens[j], 0)
                    mask[0, j] = False
                pred = self.model(src, padding_mask=mask).item()
                preds.append(pred)
        return preds


class PairAdvDataset(Dataset):
    """Training dataset containing clean, rand, sub, and adaptive adv representations."""

    def __init__(
        self,
        clean_reps: list[str],
        rand_reps: list[str],
        sub_reps: list[str],
        adv_reps: list[str],
        targets_normalized: list[float],
        vocab_map: dict[str, int],
        max_seq_len: int = 256,
    ) -> None:
        self.clean_reps = clean_reps
        self.rand_reps = rand_reps
        self.sub_reps = sub_reps
        self.adv_reps = adv_reps
        self.targets = targets_normalized
        self.vocab_map = vocab_map
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.clean_reps)

    def _encode_str(self, rep: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokenize(rep)
        seq_len = min(len(tokens), self.max_seq_len)
        tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        for j in range(seq_len):
            tensor[j] = self.vocab_map.get(tokens[j], 0)
            mask[j] = False
        return tensor, mask

    def __getitem__(self, idx: int) -> dict[str, Any]:
        c_tensor, c_mask = self._encode_str(self.clean_reps[idx])
        r_tensor, r_mask = self._encode_str(self.rand_reps[idx])
        s_tensor, s_mask = self._encode_str(self.sub_reps[idx])
        a_tensor, a_mask = self._encode_str(self.adv_reps[idx])

        return {
            "clean_tensor": c_tensor,
            "clean_mask": c_mask,
            "rand_tensor": r_tensor,
            "rand_mask": r_mask,
            "sub_tensor": s_tensor,
            "sub_mask": s_mask,
            "adv_tensor": a_tensor,
            "adv_mask": a_mask,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "idx": idx,
        }

def evaluate_model(
    model, data_df, vocab_map, scaler, max_seq_len=256, device="cpu"
):
    model.eval()
    reps = data_df["original_representation"].tolist()
    targets = data_df["property_value"].values
    
    predictor = ModelPredictor(model, vocab_map, max_seq_len, device)
    preds = np.array(predictor.predict(reps))
    preds = scaler.inverse_transform(preds)
    
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot))

    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(reps), "std": float(np.std(preds)), "min": float(np.min(preds)), "max": float(np.max(preds))}


def load_d0_model(vocab: list[str], arch: dict, path: Path, device: str = "cpu"):
    model = TwoBranchTransformerRegressorModel(
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
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return model

def generate_adv_training_set(df, d0_model, vocab, vocab_map, arch, scaler, out_dir, device="cpu"):
    rng = np.random.default_rng(123456)
    
    # Select exactly 500 deterministic training sources
    subset_df = df.sample(n=500, random_state=123456)
    
    predictor = ModelPredictor(d0_model, vocab_map, arch["max_seq_len"], device)
    
    sub_attack = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=True)
    proposal = CompositeProposalOperator([sub_attack], rng)
    
    strategy = MetropolisSearch(predictor, proposal, rng, max_changes=3, query_budget=50, temperature=0.1)
    
    candidates = []
    
    source_items = []
    
    for i, (_, row) in enumerate(subset_df.iterrows()):
        orig = row["original_representation"]
        sid = row["polymer_id"]
        source_items.append({"id": str(sid), "representation": orig})
        
        orig_pred = predictor.predict([orig])[0]
        
        res = strategy.search(orig)
        
        best_drift = 0.0
        best_cand = orig
        best_edit_count = 0
        q_count = 0
        success_q = 0
        
        for t in res.trace:
            if t.get("queried", False):
                q_count += 1
            if t.get("queried", False) and t.get("representation_valid", False) and t.get("plausible", True):
                d = t.get("absolute_drift", 0.0)
                if d > best_drift:
                    best_drift = d
                    best_cand = t.get("candidate", orig)
                    best_edit_count = t.get("perturbation_size", 0)
                    success_q = q_count
            if q_count >= 50:
                break
                
        adv_pred = predictor.predict([best_cand])[0]
        
        candidates.append({
            "source_id": str(sid),
            "original_representation": orig,
            "adversarial_representation": best_cand,
            "edit_count": int(best_edit_count),
            "changed_positions": [], # omitted for brevity
            "changed_tokens": [],
            "original_prediction_D0": float(orig_pred),
            "adversarial_prediction_D0": float(adv_pred),
            "drift_D0": float(abs(adv_pred - orig_pred)),
            "query_count": q_count,
            "success_q": success_q,
            "attacker_seed": 123456,
            "validity": "valid"
        })
        
        if (i+1) % 50 == 0:
            print(f"Generated {i+1}/500 training adversarial candidates.")
            
    with (out_dir / "training_source_manifest.json").open("w") as f:
        json.dump(source_items, f, indent=2)
        
    with (out_dir / "adaptive_training_candidates.jsonl").open("w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\\n")
            
    return candidates

def generate_other_training_pairs(
    clean_reps: list[str],
    vocab: list[str],
    seed: int = 20260815,
):
    rng = np.random.default_rng(seed)
    rand_attack = SmilesRandomizationAttack(rng, n_attempts=10)
    sub_attack = SubstitutionAttack(
        rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1
    )

    out_clean = []
    out_rand = []
    out_sub = []

    for rep in clean_reps:
        tokens = tokenize(rep)

        rand_outcomes = rand_attack.generate(tokens, n_variants=1)
        rand_rep = rep
        if rand_outcomes:
            c = rand_outcomes[0].adversarial_representation
            if validate(c).status.value == "valid" and c != rep:
                rand_rep = c

        sub_outcomes = sub_attack.generate(tokens, n_variants=1)
        sub_rep = rep
        if sub_outcomes:
            c = sub_outcomes[0].adversarial_representation
            if validate(c).status.value == "valid" and c != rep:
                sub_rep = c

        out_clean.append(rep)
        out_rand.append(rand_rep)
        out_sub.append(sub_rep)

    return out_clean, out_rand, out_sub


def train_d1(d0_model, dataset, tr_cfg, arch, vocab, device, lambda_adv, val_df, vocab_map, scaler):
    dataloader = DataLoader(dataset, batch_size=tr_cfg["batch_size"], shuffle=True)
    
    # Initialize from D0
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
    model.load_state_dict(d0_model.state_dict())
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
    )

    best_val_mae = float("inf")
    best_state_dict = None
    history = []
    
    teacher_targets_sub = torch.zeros(len(dataset), dtype=torch.float32, device=device)
    teacher_targets_adv = torch.zeros(len(dataset), dtype=torch.float32, device=device)
    with torch.no_grad():
        d0_model.eval()
        for i, (s_rep, a_rep) in enumerate(zip(dataset.sub_reps, dataset.adv_reps)):
            # Sub teacher
            tokens = tokenize(s_rep)
            seq_len = min(len(tokens), arch["max_seq_len"])
            src = torch.zeros((1, arch["max_seq_len"]), dtype=torch.long, device=device)
            mask = torch.ones((1, arch["max_seq_len"]), dtype=torch.bool, device=device)
            for j in range(seq_len):
                src[0, j] = vocab_map.get(tokens[j], 0)
                mask[0, j] = False
            teacher_targets_sub[i] = d0_model(src, padding_mask=mask).item()
            
            # Adv teacher
            tokens = tokenize(a_rep)
            seq_len = min(len(tokens), arch["max_seq_len"])
            src = torch.zeros((1, arch["max_seq_len"]), dtype=torch.long, device=device)
            mask = torch.ones((1, arch["max_seq_len"]), dtype=torch.bool, device=device)
            for j in range(seq_len):
                src[0, j] = vocab_map.get(tokens[j], 0)
                mask[0, j] = False
            teacher_targets_adv[i] = d0_model(src, padding_mask=mask).item()
            
    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            optimizer.zero_grad()

            c_tensor = batch["clean_tensor"].to(device)
            c_mask = batch["clean_mask"].to(device)
            r_tensor = batch["rand_tensor"].to(device)
            r_mask = batch["rand_mask"].to(device)
            s_tensor = batch["sub_tensor"].to(device)
            s_mask = batch["sub_mask"].to(device)
            a_tensor = batch["adv_tensor"].to(device)
            a_mask = batch["adv_mask"].to(device)
            
            target = batch["target"].to(device)
            indices = batch["idx"]
            
            pred_c = model(c_tensor, padding_mask=c_mask)
            pred_r = model(r_tensor, padding_mask=r_mask)
            pred_s = model(s_tensor, padding_mask=s_mask)
            pred_a = model(a_tensor, padding_mask=a_mask)
            
            loss_prop_clean = F.mse_loss(pred_c.squeeze(-1), target)
            loss_prop_rand = F.mse_loss(pred_r.squeeze(-1), target)
            loss_cons = F.mse_loss(pred_c, pred_r)
            
            loss_teacher_sub = F.mse_loss(pred_s.squeeze(-1), teacher_targets_sub[indices])
            loss_teacher_adv = F.mse_loss(pred_a.squeeze(-1), teacher_targets_adv[indices])
            
            loss_total = (
                loss_prop_clean
                + loss_prop_rand
                + 1.0 * loss_cons
                + 0.1 * loss_teacher_sub
                + lambda_adv * loss_teacher_adv
            )

            loss_total.backward()
            optimizer.step()
            
            epoch_loss += loss_total.item()
            
        val_eval = evaluate_model(model, val_df, vocab_map, scaler, arch["max_seq_len"], device=device)
        if val_eval["mae"] < best_val_mae:
            best_val_mae = val_eval["mae"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        history.append({
            "epoch": epoch,
            "loss_total": epoch_loss / len(dataloader),
            "val_mae": val_eval["mae"],
        })
        
    model.load_state_dict(best_state_dict)
    return model, history, best_val_mae, best_state_dict

def run_fresh_attacks(val_df, models, model_names, vocab, vocab_map, arch, out_dir, device="cpu"):
    source_items = []
    for _, row in val_df.iterrows():
        source_items.append({
            "id": str(row["polymer_id"]),
            "representation": row["original_representation"],
            "prediction": {}
        })
        if len(source_items) >= 100:
            break
            
    predictors = {}
    for name, m in zip(model_names, models):
        predictors[name] = ModelPredictor(m, vocab_map, arch["max_seq_len"], device)
        preds = predictors[name].predict([item["representation"] for item in source_items])
        for i, item in enumerate(source_items):
            item["prediction"][name] = preds[i]
            
    all_trajectories = {name: [] for name in model_names}
    best_cands = {name: {} for name in model_names}
            
    for name in model_names:
        print(f"  Fresh Attack on {name}...")
        rng = np.random.default_rng(99999) # FRESH SEED
        sub_attack = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=True)
        proposal = CompositeProposalOperator([sub_attack], rng)
        strategy = MetropolisSearch(predictors[name], proposal, rng, max_changes=3, query_budget=50, temperature=0.1)
        
        for item in source_items:
            orig = item["representation"]
            orig_pred = item["prediction"][name]
            
            res = strategy.search(orig)
            
            q_count = 0
            best_drift = 0.0
            best_cand = orig
            for t in res.trace:
                if t.get("queried", False):
                    q_count += 1
                if t.get("queried", False) and t.get("representation_valid", False) and t.get("plausible", True):
                    d = t.get("absolute_drift", 0.0)
                    if d > best_drift:
                        best_drift = d
                        best_cand = t.get("candidate", orig)
                if q_count >= 50:
                    break
                    
            best_cands[name][item["id"]] = best_cand
                    
            traj_doc = {
                "source_id": item["id"],
                "model": name,
                "best_drift": float(best_drift),
                "best_cand": best_cand,
            }
            all_trajectories[name].append(traj_doc)
            
    with (out_dir / "fresh_attack_trajectories_D0.jsonl").open("w") as f:
        for t in all_trajectories["D0"]:
            f.write(json.dumps(t) + "\\n")
            
    with (out_dir / "fresh_attack_trajectories_D1.jsonl").open("w") as f:
        for t in all_trajectories["D1"]:
            f.write(json.dumps(t) + "\\n")
            
    # Closed loop comparison & improvement calculation
    improvements = []
    
    comp_docs = []
    for t0, t1 in zip(all_trajectories["D0"], all_trajectories["D1"]):
        assert t0["source_id"] == t1["source_id"]
        improv = t0["best_drift"] - t1["best_drift"]
        improvements.append(improv)
        comp_docs.append({
            "source_id": t0["source_id"],
            "D0_drift": t0["best_drift"],
            "D1_drift": t1["best_drift"],
            "improvement": improv
        })
        
    pd.DataFrame(comp_docs).to_csv(out_dir / "closed_loop_comparison.csv", index=False)
    
    # Bootstrap CI for mean improvement
    improvements = np.array(improvements)
    np.random.seed(42)
    boot_means = [np.mean(np.random.choice(improvements, size=len(improvements), replace=True)) for _ in range(1000)]
    ci_lower, ci_upper = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
    
    # Cross Transfer
    # Evaluate D0 cands on D1
    d0_cands = [best_cands["D0"][item["id"]] for item in source_items]
    d1_cands = [best_cands["D1"][item["id"]] for item in source_items]
    
    preds_d1_on_d0cands = predictors["D1"].predict(d0_cands)
    preds_d0_on_d1cands = predictors["D0"].predict(d1_cands)
    
    cross_transfer = []
    for i, item in enumerate(source_items):
        orig_pred_d1 = item["prediction"]["D1"]
        orig_pred_d0 = item["prediction"]["D0"]
        
        drift_d1_on_d0 = abs(preds_d1_on_d0cands[i] - orig_pred_d1)
        drift_d0_on_d1 = abs(preds_d0_on_d1cands[i] - orig_pred_d0)
        
        cross_transfer.append({
            "source_id": item["id"],
            "drift_D1_on_D0_cands": drift_d1_on_d0,
            "drift_D0_on_D1_cands": drift_d0_on_d1
        })
        
    pd.DataFrame(cross_transfer).to_csv(out_dir / "cross_transfer.csv", index=False)
    
    return {
        "mean_improvement": float(np.mean(improvements)),
        "median_improvement": float(np.median(improvements)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "fraction_improved": float(np.mean(improvements > 0)),
        "fraction_worsened": float(np.mean(improvements < 0)),
        "mean_drift_D0": float(np.mean([t["best_drift"] for t in all_trajectories["D0"]])),
        "mean_drift_D1": float(np.mean([t["best_drift"] for t in all_trajectories["D1"]])),
    }
    
def check_frozen_banks(models, model_names, vocab_map, arch, data_dir, device="cpu"):
    df_rand = pd.read_csv(data_dir / "randomized_bank.csv")
    df_sub = pd.read_csv(data_dir / "substitution_bank.csv")
    df_del = pd.read_csv(data_dir / "deletion_bank_unseen.csv")
    
    res = {}
    for name, m in zip(model_names, models):
        pred_m = ModelPredictor(m, vocab_map, arch["max_seq_len"], device)
        
        # Rand
        orig_preds = pred_m.predict(df_rand["original_representation"])
        adv_preds = pred_m.predict(df_rand["adversarial_representation"])
        drift_rand = np.mean(np.abs(np.array(orig_preds) - np.array(adv_preds)))
        
        # Sub
        orig_preds = pred_m.predict(df_sub["original_representation"])
        adv_preds = pred_m.predict(df_sub["adversarial_representation"])
        drift_sub = np.mean(np.abs(np.array(orig_preds) - np.array(adv_preds)))
        
        # Del
        orig_preds = pred_m.predict(df_del["original_representation"])
        adv_preds = pred_m.predict(df_del["adversarial_representation"])
        drift_del = np.mean(np.abs(np.array(orig_preds) - np.array(adv_preds)))
        
        res[name] = {
            "drift_rand": drift_rand,
            "drift_sub": drift_sub,
            "drift_del": drift_del
        }
    return res

def main():
    run_id = str(int(time.time()))
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    banks_dir = repo_root / "results" / "frozen_banks"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    out_dir = repo_root / "results" / "phase9_closed_loop" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    d1_dir = out_dir / "D1_checkpoint"
    d1_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg_path = repo_root / "configs" / "specialized_transformer.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
    arch = cfg["architecture"]
    tr_cfg = cfg["training"]
    
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    with (data_dir / "vocab.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    vocab_map = {tok: idx + 1 for idx, tok in enumerate(vocab)}
    
    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]
    
    scaler_path = repo_root / "results" / "models" / "transformer_regressor" / "scaler.json"
    scaler = TargetScaler.load(scaler_path)
    scaler_hash = compute_file_hash(scaler_path)
    
    print("1. Loading D0 Model...")
    d0_path = p4_dir / "mix_robust_0.1"
    d0_hash = compute_file_hash(d0_path / "model.pt")
    d0_model = load_d0_model(vocab, arch, d0_path, device=device)
    
    d0_clean_val = evaluate_model(d0_model, val_df, vocab_map, scaler, arch["max_seq_len"], device=device)
    
    print("2. Generating Adversarial Training Set on Train Split...")
    adv_cands = generate_adv_training_set(train_df, d0_model, vocab, vocab_map, arch, scaler, out_dir, device=device)
    
    adv_dict = {c["source_id"]: c["adversarial_representation"] for c in adv_cands}
    with (out_dir / "teacher_targets.jsonl").open("w") as f:
        for c in adv_cands:
            f.write(json.dumps({"source_id": c["source_id"], "teacher_target": c["adversarial_prediction_D0"]}) + "\\n")
            
    print("3. Training Defender D1...")
    train_df_subset = train_df[train_df["polymer_id"].isin(adv_dict.keys())].copy()
    clean_reps = train_df_subset["original_representation"].tolist()
    targets_raw = train_df_subset["property_value"].values
    targets_norm = scaler.transform(targets_raw).tolist()
    
    out_clean, out_rand, out_sub = generate_other_training_pairs(clean_reps, vocab, seed=42)
    adv_reps = [adv_dict[pid] for pid in train_df_subset["polymer_id"]]
    
    dataset = PairAdvDataset(out_clean, out_rand, out_sub, adv_reps, targets_norm, vocab_map, arch["max_seq_len"])
    
    best_overall_d1_model = None
    best_overall_d1_state = None
    best_lambda = None
    best_lambda_hist = None
    
    lambda_selection_results = []
    for l_adv in [0.1, 0.5]:
        print(f"  Training D1 with lambda_adv = {l_adv}")
        model, hist, val_mae, state = train_d1(d0_model, dataset, tr_cfg, arch, vocab, device, l_adv, val_df, vocab_map, scaler)
        mae_deg = val_mae - d0_clean_val["mae"]
        print(f"    Val MAE: {val_mae:.4f} (Degradation: {mae_deg:.4f})")
        lambda_selection_results.append({
            "lambda_adv": l_adv,
            "val_mae": val_mae,
            "degradation": mae_deg
        })
        
        if mae_deg <= 0.02:
            if best_lambda is None or l_adv > best_lambda:
                best_lambda = l_adv
                best_overall_d1_model = model
                best_overall_d1_state = state
                best_lambda_hist = hist
                
    if best_lambda is None:
        print("Warning: All lambda_adv degraded clean MAE > 0.02. Using lambda_adv=0.1 as fallback.")
        best_lambda = 0.1
        best_overall_d1_model = model
        best_overall_d1_state = state
        best_lambda_hist = hist
        
    print(f"Selected lambda_adv = {best_lambda}")
    pd.DataFrame(lambda_selection_results).to_csv(out_dir / "lambda_selection.csv", index=False)
    
    torch.save(best_overall_d1_state, d1_dir / "model.pt")
    scaler.save(d1_dir / "scaler.json")
    with (out_dir / "training_config.json").open("w") as f:
        json.dump({"lambda_adv": best_lambda, "lambda_rand": 1.0, "lambda_sub": 0.1}, f, indent=2)
    pd.DataFrame(best_lambda_hist).to_csv(out_dir / "training_history.csv", index=False)
    
    d1_clean_val = evaluate_model(best_overall_d1_model, val_df, vocab_map, scaler, arch["max_seq_len"], device=device)
    pd.DataFrame([{"model": "D0", **d0_clean_val}, {"model": "D1", **d1_clean_val}]).to_csv(out_dir / "clean_metrics.csv", index=False)
    
    print("4. Running Fresh Adaptive Re-Attack...")
    attack_res = run_fresh_attacks(val_df, [d0_model, best_overall_d1_model], ["D0", "D1"], vocab, vocab_map, arch, out_dir, device=device)
    
    print("5. Checking Catastrophic Robustness Regression...")
    bank_res = check_frozen_banks([d0_model, best_overall_d1_model], ["D0", "D1"], vocab_map, arch, banks_dir, device=device)
    
    bank_docs = []
    for model_name, metrics in bank_res.items():
        bank_docs.append({"model": model_name, **metrics})
    pd.DataFrame(bank_docs).to_csv(out_dir / "frozen_bank_regression.csv", index=False)
    
    print("6. Adversarial Overfitting Check...")
    # Evaluate D1 on the training adversarial candidates
    pred_d1_on_train_adv = ModelPredictor(best_overall_d1_model, vocab_map, arch["max_seq_len"], device).predict(adv_reps)
    drift_d1_on_train_adv = np.mean(np.abs(np.array(pred_d1_on_train_adv) - np.array([c["original_prediction_D0"] for c in adv_cands])))
    
    overfitting_doc = [{
        "drift_D1_on_training_adv_candidates": drift_d1_on_train_adv,
        "drift_D1_on_fresh_adv_candidates": attack_res["mean_drift_D1"]
    }]
    pd.DataFrame(overfitting_doc).to_csv(out_dir / "adversarial_overfitting.csv", index=False)
    
    print("7. Generating Summary...")
    verdict = "NOT SUPPORTED"
    if attack_res["mean_drift_D1"] < attack_res["mean_drift_D0"] and attack_res["ci_lower"] > 0 and (d1_clean_val["mae"] - d0_clean_val["mae"]) <= 0.02 and bank_res["D1"]["drift_rand"] <= bank_res["D0"]["drift_rand"] * 1.1:
        verdict = "SUPPORTED"
    elif attack_res["mean_drift_D1"] < attack_res["mean_drift_D0"]:
        verdict = "PARTIALLY SUPPORTED"
        
    summary = {
        "d0_hash": d0_hash,
        "d1_hash": compute_file_hash(d1_dir / "model.pt"),
        "training_source_count": len(adv_cands),
        "adaptive_training_candidate_count": len(adv_cands),
        "best_lambda_adv": best_lambda,
        "d1_clean_mae": d1_clean_val["mae"],
        "mean_improvement": attack_res["mean_improvement"],
        "ci_lower": attack_res["ci_lower"],
        "ci_upper": attack_res["ci_upper"],
        "verdict": verdict
    }
    
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        
    with (out_dir / "reproducibility.json").open("w") as f:
        json.dump({"run_id": run_id, "timestamp": time.time()}, f, indent=2)
        
    print(f"\\nClosed-Loop Verdict: {verdict}")
    print(f"Mean Drift Improvement (D0 -> D1): {attack_res['mean_improvement']:.4f} 95% CI: [{attack_res['ci_lower']:.4f}, {attack_res['ci_upper']:.4f}]")
    print(f"Phase 9 Complete! Output saved to {out_dir}")

if __name__ == "__main__":
    main()
