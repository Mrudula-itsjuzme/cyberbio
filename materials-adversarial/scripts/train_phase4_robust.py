#!/usr/bin/env python3
"""Phase 4: Controlled Robustness Training.

Trains:
- Randomization-Robust Model
- Mixed Robust Model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from materials_adv.domain.chemistry.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.specialized_transformer import (
    TwoBranchTransformerRegressorModel,
)
from materials_adv.validation.pipeline import validate


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class PairDataset(Dataset):
    """Training dataset containing (clean, randomized, substituted) representations and clean targets."""

    def __init__(
        self,
        clean_reps: list[str],
        rand_reps: list[str],
        sub_reps: list[str],
        targets_normalized: list[float],
        vocab_map: dict[str, int],
        max_seq_len: int = 256,
    ) -> None:
        self.clean_reps = clean_reps
        self.rand_reps = rand_reps
        self.sub_reps = sub_reps
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

        return {
            "clean_tensor": c_tensor,
            "clean_mask": c_mask,
            "rand_tensor": r_tensor,
            "rand_mask": r_mask,
            "sub_tensor": s_tensor,
            "sub_mask": s_mask,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "idx": idx,
        }


def generate_training_pairs(
    clean_reps: list[str],
    vocab: list[str],
    seed: int = 20260815,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Generate training pairs for train split polymers."""
    rng = np.random.default_rng(seed)
    rand_attack = SmilesRandomizationAttack(rng, n_attempts=10)
    sub_attack = SubstitutionAttack(
        rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1
    )

    out_clean = []
    out_rand = []
    out_sub = []
    provenance = []

    counts = {"total_sources": len(clean_reps), "rand_valid": 0, "sub_valid": 0}

    for source_id, rep in enumerate(clean_reps):
        tokens = tokenize(rep)

        # 1. Generate SMILES Randomization pair
        rand_outcomes = rand_attack.generate(tokens, n_variants=1)
        rand_rep = rep
        if rand_outcomes:
            candidate_rand = rand_outcomes[0].adversarial_representation
            val = validate(candidate_rand)
            if val.status.value == "valid" and candidate_rand != rep:
                rand_rep = candidate_rand
                counts["rand_valid"] += 1
                provenance.append({
                    "source_id": source_id,
                    "candidate_id": f"train_rand_{source_id}",
                    "attack_family": "randomization",
                    "original_representation": rep,
                    "candidate_representation": rand_rep,
                    "canonical_equivalent": True,
                    "label_basis": "measured_physical_supervision"
                })

        # 2. Generate Substitution pair
        sub_outcomes = sub_attack.generate(tokens, n_variants=1)
        sub_rep = rep
        if sub_outcomes:
            candidate_sub = sub_outcomes[0].adversarial_representation
            val = validate(candidate_sub)
            if val.status.value == "valid" and candidate_sub != rep:
                sub_rep = candidate_sub
                counts["sub_valid"] += 1
                provenance.append({
                    "source_id": source_id,
                    "candidate_id": f"train_sub_{source_id}",
                    "attack_family": "substitution",
                    "original_representation": rep,
                    "candidate_representation": sub_rep,
                    "canonical_equivalent": False,
                    "label_basis": "clean_model_teacher"
                })

        out_clean.append(rep)
        out_rand.append(rand_rep)
        out_sub.append(sub_rep)

    return out_clean, out_rand, out_sub, {"counts": counts, "provenance": provenance}


def load_teacher_model(vocab: list[str], arch: dict, path: Path, device: str = "cpu") -> TwoBranchTransformerRegressorModel:
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


def evaluate_model(
    model: TwoBranchTransformerRegressorModel,
    data_df: pd.DataFrame,
    vocab_map: dict[str, int],
    scaler: TargetScaler,
    max_seq_len: int = 256,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    reps = data_df["original_representation"].tolist()
    targets = data_df["property_value"].values

    preds = []
    with torch.no_grad():
        for rep in reps:
            tokens = tokenize(rep)
            seq_len = min(len(tokens), max_seq_len)
            src = torch.zeros((1, max_seq_len), dtype=torch.long, device=device)
            mask = torch.ones((1, max_seq_len), dtype=torch.bool, device=device)
            for j in range(seq_len):
                src[0, j] = vocab_map.get(tokens[j], 0)
                mask[0, j] = False
            norm_pred = model(src, padding_mask=mask).item()
            unscaled = scaler.inverse_transform(np.array([norm_pred]))[0]
            preds.append(unscaled)

    preds = np.array(preds)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot))

    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(reps)}


def train_model_variant(
    mode: str,
    output_dir: Path,
    lambda_consistency: float,
    lambda_teacher: float,
    seed: int = 20260815,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "results" / "models"
    clean_dir = models_dir / "transformer_regressor"
    control_dir = models_dir / "specialized_control"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load config
    cfg_path = repo_root / "configs" / "specialized_transformer.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)

    arch = cfg["architecture"]
    tr_cfg = cfg["training"]

    # 2. Load dataset & splits
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    with (data_dir / "vocab.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_map = {tok: idx + 1 for idx, tok in enumerate(vocab)}

    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]
    test_df = df.iloc[splits["test"]]

    # 3. Load frozen clean scaler and teacher model
    scaler_path = clean_dir / "scaler.json"
    scaler = TargetScaler.load(scaler_path)
    scaler_hash = compute_file_hash(scaler_path)
    
    teacher_model = None
    teacher_hash = None
    if mode == "mixed_robust":
        teacher_model = load_teacher_model(vocab, arch, control_dir, device=device)
        teacher_hash = compute_file_hash(control_dir / "model.pt")

    # 4. Generate training pairs
    train_clean_reps = train_df["original_representation"].tolist()
    train_targets_raw = train_df["property_value"].values
    train_targets_norm = scaler.transform(train_targets_raw).tolist()

    c_reps, r_reps, s_reps, meta = generate_training_pairs(
        train_clean_reps, vocab, seed=seed
    )
    pair_counts = meta["counts"]
    provenance = meta["provenance"]
    
    if mode == "mixed_robust":
        for p in provenance:
            if p["attack_family"] == "substitution":
                p["teacher_checkpoint_hash"] = teacher_hash

    # Save provenance
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    dataset = PairDataset(
        c_reps, r_reps, s_reps, train_targets_norm, vocab_map, arch["max_seq_len"]
    )
    dataloader = DataLoader(dataset, batch_size=tr_cfg["batch_size"], shuffle=True)

    # 5. Initialize Model
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
    )

    best_val_mae = float("inf")
    best_epoch = 0
    best_state_dict = None
    history = []

    # Pre-compute teacher targets for substitution to save time (if mixed_robust)
    teacher_targets_sub = None
    if teacher_model is not None:
        teacher_targets_sub = torch.zeros(len(dataset), dtype=torch.float32, device=device)
        with torch.no_grad():
            for i, rep in enumerate(s_reps):
                tokens = tokenize(rep)
                seq_len = min(len(tokens), arch["max_seq_len"])
                src = torch.zeros((1, arch["max_seq_len"]), dtype=torch.long, device=device)
                mask = torch.ones((1, arch["max_seq_len"]), dtype=torch.bool, device=device)
                for j in range(seq_len):
                    src[0, j] = vocab_map.get(tokens[j], 0)
                    mask[0, j] = False
                teacher_targets_sub[i] = teacher_model(src, padding_mask=mask).item()

    # 6. Training Loop
    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_prop_clean = 0.0
        epoch_prop_rand = 0.0
        epoch_cons = 0.0
        epoch_teacher = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            c_tensor = batch["clean_tensor"].to(device)
            c_mask = batch["clean_mask"].to(device)
            r_tensor = batch["rand_tensor"].to(device)
            r_mask = batch["rand_mask"].to(device)
            
            target = batch["target"].to(device)
            indices = batch["idx"]

            pred_c = model(c_tensor, padding_mask=c_mask)
            pred_r = model(r_tensor, padding_mask=r_mask)

            # A. Property Loss (Clean and Rand)
            loss_prop_clean = F.mse_loss(pred_c.squeeze(-1), target)
            loss_prop_rand = F.mse_loss(pred_r.squeeze(-1), target)

            # B. Consistency Loss
            loss_cons = F.mse_loss(pred_c, pred_r)

            # C. Teacher Loss (Substitution)
            loss_teacher_val = torch.tensor(0.0, device=device)
            if mode == "mixed_robust" and teacher_model is not None:
                s_tensor = batch["sub_tensor"].to(device)
                s_mask = batch["sub_mask"].to(device)
                pred_s = model(s_tensor, padding_mask=s_mask)
                batch_teacher_targets = teacher_targets_sub[indices]
                loss_teacher_val = F.mse_loss(pred_s.squeeze(-1), batch_teacher_targets)

            loss_total = (
                loss_prop_clean
                + loss_prop_rand
                + lambda_consistency * loss_cons
                + lambda_teacher * loss_teacher_val
            )

            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item()
            epoch_prop_clean += loss_prop_clean.item()
            epoch_prop_rand += loss_prop_rand.item()
            epoch_cons += loss_cons.item()
            if mode == "mixed_robust" and teacher_model is not None:
                epoch_teacher += loss_teacher_val.item()

        # Validation evaluation for early stopping / best epoch selection
        val_eval = evaluate_model(model, val_df, vocab_map, scaler, arch["max_seq_len"], device=device)
        if val_eval["mae"] < best_val_mae:
            best_val_mae = val_eval["mae"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "loss_total": epoch_loss / len(dataloader),
                "loss_prop_clean": epoch_prop_clean / len(dataloader),
                "loss_prop_rand": epoch_prop_rand / len(dataloader),
                "loss_cons": epoch_cons / len(dataloader),
                "loss_teacher": epoch_teacher / len(dataloader),
                "val_mae": val_eval["mae"],
            }
        )

    # Load best checkpoint from validation selection
    model.load_state_dict(best_state_dict)

    # 7. Final Evaluations (Validation & Exposed Test Reference)
    val_final = evaluate_model(model, val_df, vocab_map, scaler, arch["max_seq_len"], device=device)
    test_final = evaluate_model(model, test_df, vocab_map, scaler, arch["max_seq_len"], device=device)

    # Save artifact
    torch.save(best_state_dict, output_dir / "model.pt")
    scaler.save(output_dir / "scaler.json")

    metrics_payload = {
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mae": val_final["mae"],
        "val_rmse": val_final["rmse"],
        "val_r2": val_final["r2"],
        "exposed_test_reference": {
            "test_mae": test_final["mae"],
            "test_rmse": test_final["rmse"],
            "test_r2": test_final["r2"],
            "note": "EXPOSED TEST REFERENCE ONLY; NOT USED FOR MODEL SELECTION",
        },
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture": arch,
        "loss_hyperparameters": {
            "lambda_consistency": lambda_consistency,
            "lambda_teacher": lambda_teacher,
        },
        "pair_counts": pair_counts,
        "model_hash": compute_file_hash(output_dir / "model.pt"),
        "scaler_hash": compute_file_hash(output_dir / "scaler.json"),
        "teacher_hash": teacher_hash,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Mode {mode!r} training completed (λ_cons={lambda_consistency}, λ_teach={lambda_teacher}).")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Validation MAE: {val_final['mae']:.4f} eV")
    print(f"  Exposed Test MAE (Ref): {test_final['mae']:.4f} eV")
    print(f"  Saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Train Robust Two-Branch Transformer Models")
    parser.add_argument(
        "--mode",
        choices=["randomization_robust", "mixed_robust"],
        required=True,
        help="Training mode",
    )
    parser.add_argument("--lambda-consistency", type=float, default=0.0, help="Weight for randomization consistency loss")
    parser.add_argument("--lambda-teacher", type=float, default=0.0, help="Weight for substitution teacher loss")
    parser.add_argument(
        "--seed", type=int, default=20260815, help="Random training seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Target output directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / args.output_dir

    train_model_variant(
        mode=args.mode,
        output_dir=out_path,
        lambda_consistency=args.lambda_consistency,
        lambda_teacher=args.lambda_teacher,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
