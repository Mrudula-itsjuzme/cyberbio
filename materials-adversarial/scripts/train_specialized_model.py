#!/usr/bin/env python3
"""Train Two-Branch Specialized Transformer Model & Architecture Control.

Trains two model variants on polyVERSE Bandgap (eV):
1. Architecture Control Model: Two-branch architecture trained ONLY on clean property regression.
2. Specialized Two-Branch Model: Two-branch architecture trained with multi-objective auxiliary losses
   (representation invariance, chemistry sensitivity, and branch de-collapse).

Model selection is strictly guided by Validation MAE (N=631). Exposed test metrics (N=632)
are calculated ONLY after model selection is frozen and labeled as EXPOSED TEST REFERENCE.
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
    TwoBranchTransformerRegressor,
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
    """Training dataset containing (clean, randomized_equivalent, substituted) representations and clean targets."""

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
        }


def generate_training_pairs(
    clean_reps: list[str],
    vocab: list[str],
    seed: int = 20260815,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Generate training pairs for train split polymers."""
    rng = np.random.default_rng(seed)
    rand_attack = SmilesRandomizationAttack(rng, n_attempts=10)
    sub_attack = SubstitutionAttack(
        rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1
    )

    out_clean = []
    out_rand = []
    out_sub = []

    counts = {"total_sources": len(clean_reps), "rand_valid": 0, "sub_valid": 0}

    for rep in clean_reps:
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

        # 2. Generate Substitution pair
        sub_outcomes = sub_attack.generate(tokens, n_variants=1)
        sub_rep = rep
        if sub_outcomes:
            candidate_sub = sub_outcomes[0].adversarial_representation
            val = validate(candidate_sub)
            if val.status.value == "valid" and candidate_sub != rep:
                sub_rep = candidate_sub
                counts["sub_valid"] += 1

        out_clean.append(rep)
        out_rand.append(rand_rep)
        out_sub.append(sub_rep)

    return out_clean, out_rand, out_sub, counts


def compute_auxiliary_losses(
    z_clean_a: torch.Tensor,
    z_clean_b: torch.Tensor,
    z_rand_a: torch.Tensor,
    z_sub_b: torch.Tensor,
    chem_margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (loss_repr, loss_chem, loss_div)."""
    # A. Representation Invariance Loss (Branch A): 1 - cos(z_clean_a, z_rand_a)
    cos_a = F.cosine_similarity(z_clean_a, z_rand_a, dim=-1)
    loss_repr = torch.mean(1.0 - cos_a)

    # B. Chemistry Sensitivity Loss (Branch B): Margin-based L2 distance
    dist_b = torch.norm(z_clean_b - z_sub_b, p=2, dim=-1)
    loss_chem = torch.mean(F.relu(chem_margin - dist_b))

    # C. Specialization / De-collapse Loss: Abs Cosine similarity between Branch A & B
    cos_ab = F.cosine_similarity(z_clean_a, z_clean_b, dim=-1)
    loss_div = torch.mean(torch.abs(cos_ab))

    return loss_repr, loss_chem, loss_div


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


def compute_branch_diagnostics(
    model: TwoBranchTransformerRegressorModel,
    data_df: pd.DataFrame,
    vocab_map: dict[str, int],
    vocab: list[str],
    seed: int = 20260815,
    max_seq_len: int = 256,
    device: str = "cpu",
) -> dict[str, float]:
    model.eval()
    reps = data_df["original_representation"].tolist()

    rng = np.random.default_rng(seed)
    rand_attack = SmilesRandomizationAttack(rng, n_attempts=10)
    sub_attack = SubstitutionAttack(
        rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1
    )

    z_a_clean_list, z_b_clean_list = [], []
    dist_a_rand, dist_b_rand = [], []
    dist_a_sub, dist_b_sub = [], []

    with torch.no_grad():
        for rep in reps[:200]:  # Use first 200 samples for diagnostics
            tokens = tokenize(rep)

            def get_zb(r_str):
                tks = tokenize(r_str)
                slen = min(len(tks), max_seq_len)
                s = torch.zeros((1, max_seq_len), dtype=torch.long, device=device)
                m = torch.ones((1, max_seq_len), dtype=torch.bool, device=device)
                for j in range(slen):
                    s[0, j] = vocab_map.get(tks[j], 0)
                    m[0, j] = False
                _, za, zb = model.forward_branches(s, padding_mask=m)
                return za[0].cpu().numpy(), zb[0].cpu().numpy()

            za_c, zb_c = get_zb(rep)
            z_a_clean_list.append(za_c)
            z_b_clean_list.append(zb_c)

            # Randomization diagnostic
            r_outs = rand_attack.generate(tokens, n_variants=1)
            if r_outs and validate(r_outs[0].adversarial_representation).status.value == "valid":
                r_rep = r_outs[0].adversarial_representation
                if r_rep != rep:
                    za_r, zb_r = get_zb(r_rep)
                    dist_a_rand.append(np.linalg.norm(za_c - za_r))
                    dist_b_rand.append(np.linalg.norm(zb_c - zb_r))

            # Substitution diagnostic
            s_outs = sub_attack.generate(tokens, n_variants=1)
            if s_outs and validate(s_outs[0].adversarial_representation).status.value == "valid":
                s_rep = s_outs[0].adversarial_representation
                if s_rep != rep:
                    za_s, zb_s = get_zb(s_rep)
                    dist_a_sub.append(np.linalg.norm(za_c - za_s))
                    dist_b_sub.append(np.linalg.norm(zb_c - zb_s))

    z_a_arr = np.array(z_a_clean_list)
    z_b_arr = np.array(z_b_clean_list)

    norm_a = float(np.mean(np.linalg.norm(z_a_arr, axis=1)))
    norm_b = float(np.mean(np.linalg.norm(z_b_arr, axis=1)))
    var_a = float(np.mean(np.var(z_a_arr, axis=0)))
    var_b = float(np.mean(np.var(z_b_arr, axis=0)))

    # Cosine similarity between Branch A and B
    cos_ab = np.mean(
        [
            np.dot(za, zb) / (np.linalg.norm(za) * np.linalg.norm(zb) + 1e-9)
            for za, zb in zip(z_a_arr, z_b_arr)
        ]
    )

    return {
        "norm_branch_a": norm_a,
        "norm_branch_b": norm_b,
        "var_branch_a": var_a,
        "var_branch_b": var_b,
        "cos_sim_branch_a_b": float(cos_ab),
        "mean_dist_a_rand": float(np.mean(dist_a_rand)) if dist_a_rand else 0.0,
        "mean_dist_b_rand": float(np.mean(dist_b_rand)) if dist_b_rand else 0.0,
        "mean_dist_a_sub": float(np.mean(dist_a_sub)) if dist_a_sub else 0.0,
        "mean_dist_b_sub": float(np.mean(dist_b_sub)) if dist_b_sub else 0.0,
    }


def train_model_variant(
    mode: str,
    output_dir: Path,
    seed: int = 20260815,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "results" / "models"
    clean_dir = models_dir / "transformer_regressor"

    # 1. Load config
    cfg_path = repo_root / "configs" / "specialized_transformer.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        import yaml

        cfg = yaml.safe_load(f)

    arch = cfg["architecture"]
    l_hyp = cfg["loss_hyperparameters"]
    tr_cfg = cfg["training"]

    if mode == "control":
        lambda_repr = 0.0
        lambda_chem = 0.0
        lambda_div = 0.0
    else:
        lambda_repr = l_hyp["lambda_repr"]
        lambda_chem = l_hyp["lambda_chem"]
        lambda_div = l_hyp["lambda_div"]

    chem_margin = l_hyp["chem_margin"]

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

    # 3. Load frozen clean scaler
    scaler = TargetScaler.load(clean_dir / "scaler.json")

    # 4. Generate training pairs
    train_clean_reps = train_df["original_representation"].tolist()
    train_targets_raw = train_df["property_value"].values
    train_targets_norm = scaler.transform(train_targets_raw).tolist()

    c_reps, r_reps, s_reps, pair_counts = generate_training_pairs(
        train_clean_reps, vocab, seed=seed
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
    )

    best_val_mae = float("inf")
    best_epoch = 0
    best_state_dict = None
    history = []

    # 6. Training Loop
    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_prop = 0.0
        epoch_repr = 0.0
        epoch_chem = 0.0
        epoch_div = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            c_tensor = batch["clean_tensor"]
            c_mask = batch["clean_mask"]
            r_tensor = batch["rand_tensor"]
            r_mask = batch["rand_mask"]
            s_tensor = batch["sub_tensor"]
            s_mask = batch["sub_mask"]
            target = batch["target"]

            # Forward passes
            pred_c, za_c, zb_c = model.forward_branches(c_tensor, padding_mask=c_mask)
            pred_r, za_r, zb_r = model.forward_branches(r_tensor, padding_mask=r_mask)
            _, za_s, zb_s = model.forward_branches(s_tensor, padding_mask=s_mask)

            # A. Property Loss
            loss_prop = F.mse_loss(pred_c, target) + F.mse_loss(pred_r, target)

            # B. Auxiliary Losses
            loss_repr, loss_chem, loss_div = compute_auxiliary_losses(
                za_c, zb_c, za_r, zb_s, chem_margin=chem_margin
            )

            loss_total = (
                loss_prop
                + lambda_repr * loss_repr
                + lambda_chem * loss_chem
                + lambda_div * loss_div
            )

            loss_total.backward()
            optimizer.step()

            epoch_loss += loss_total.item()
            epoch_prop += loss_prop.item()
            epoch_repr += loss_repr.item()
            epoch_chem += loss_chem.item()
            epoch_div += loss_div.item()

        # Validation evaluation for early stopping / best epoch selection
        val_eval = evaluate_model(model, val_df, vocab_map, scaler, arch["max_seq_len"])
        if val_eval["mae"] < best_val_mae:
            best_val_mae = val_eval["mae"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "loss_total": epoch_loss / len(dataloader),
                "loss_prop": epoch_prop / len(dataloader),
                "loss_repr": epoch_repr / len(dataloader),
                "loss_chem": epoch_chem / len(dataloader),
                "loss_div": epoch_div / len(dataloader),
                "val_mae": val_eval["mae"],
            }
        )

    # Load best checkpoint from validation selection
    model.load_state_dict(best_state_dict)

    # 7. Final Evaluations (Validation & Exposed Test Reference)
    val_final = evaluate_model(model, val_df, vocab_map, scaler, arch["max_seq_len"])
    test_final = evaluate_model(model, test_df, vocab_map, scaler, arch["max_seq_len"])
    diagnostics = compute_branch_diagnostics(
        model, val_df, vocab_map, vocab, seed=seed, max_seq_len=arch["max_seq_len"]
    )

    # Save artifact
    output_dir.mkdir(parents=True, exist_ok=True)
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
            "lambda_repr": lambda_repr,
            "lambda_chem": lambda_chem,
            "lambda_div": lambda_div,
            "chem_margin": chem_margin,
        },
        "pair_counts": pair_counts,
        "branch_diagnostics": diagnostics,
        "model_hash": compute_file_hash(output_dir / "model.pt"),
        "scaler_hash": compute_file_hash(output_dir / "scaler.json"),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Mode {mode!r} training completed.")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Validation MAE: {val_final['mae']:.4f} eV")
    print(f"  Exposed Test MAE (Ref): {test_final['mae']:.4f} eV")
    print(f"  Saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Two-Branch Transformer Models")
    parser.add_argument(
        "--mode",
        choices=["control", "specialized"],
        required=True,
        help="Training mode: 'control' or 'specialized'",
    )
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

    train_model_variant(mode=args.mode, output_dir=out_path, seed=args.seed)


if __name__ == "__main__":
    main()
