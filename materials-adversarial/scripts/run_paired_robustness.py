#!/usr/bin/env python3
"""Phase 2 — Controlled Paired Adversarial Baseline & Defense Benchmark Runner.

Trains Defense A (Randomization) and Defense B (Teacher-Labeled Stress) models
under strict label policy semantics, then evaluates Clean Baseline, Defense A,
and Defense B on an identical, model-independent candidate bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Any

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.domain.chemistry.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.experiments.paired_robustness import (
    PairedCandidate,
    PredictionRecord,
    canonicalize_smiles,
    compute_paired_statistics,
    generate_paired_candidate_bank,
    is_canonically_equivalent,
    score_model_on_candidate_bank,
)
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.training.train import train as train_model
from materials_adv.utils.io import to_jsonable


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit(repo_root: Path) -> str:
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def load_predictor(model_dir: Path, vocab: list[str], target_units: str = "eV") -> TransformerRegressor:
    with (model_dir / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    arch = metrics.get("architecture", {})
    model = TransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=arch.get("d_model", 64),
        n_layers=arch.get("n_layers", 2),
        n_heads=arch.get("n_heads", 4),
        dim_feedforward=arch.get("dim_feedforward", 128),
        dropout=arch.get("dropout", 0.1),
        max_seq_len=arch.get("max_seq_len", 256),
        pooling=arch.get("pooling", "mean"),
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    scaler = TargetScaler.load(model_dir / "scaler.json")
    return TransformerRegressor(model=model, vocab=vocab, scaler=scaler, target_units=target_units)


def build_defense_a_dataset(
    train_df: pd.DataFrame,
    vocab: list[str],
    seed: int,
    output_csv: Path,
) -> None:
    """Build Defense A (Randomization) training set with measured_inherited_equivalent labels."""
    rng = np.random.default_rng(seed)
    attack = SmilesRandomizationAttack(rng, n_attempts=10)

    rows: list[dict[str, Any]] = []

    for idx, row in train_df.iterrows():
        source_idx = int(row.name if isinstance(row.name, int) else idx)
        orig_rep = str(row["original_representation"])
        target_val = float(row["property_value"])

        # Base clean row
        rows.append({
            "source_index": source_idx,
            "original_representation": orig_rep,
            "adversarial_representation": orig_rep,
            "attack_family": "clean",
            "canonical_equivalent": True,
            "label_basis": "measured",
            "target_value": target_val,
            "teacher_model_hash": None,
            "property_value": target_val,
        })

        # Randomized representations
        tokens = tokenize(orig_rep)
        for outcome in attack.generate(tokens, n_variants=3):
            adv_rep = outcome.adversarial_representation
            is_eq = is_canonically_equivalent(orig_rep, adv_rep)
            if is_eq:
                rows.append({
                    "source_index": source_idx,
                    "original_representation": orig_rep,
                    "adversarial_representation": adv_rep,
                    "attack_family": "randomization",
                    "canonical_equivalent": True,
                    "label_basis": "measured_inherited_equivalent",
                    "target_value": target_val,
                    "teacher_model_hash": None,
                    "property_value": target_val,
                })

    aug_df = pd.DataFrame(rows)
    # Ensure standard schema expected by train_model
    aug_df["original_representation"] = aug_df["adversarial_representation"]
    aug_df.to_csv(output_csv, index=False)
    print(f"Defense A dataset built: {len(aug_df)} rows saved to {output_csv}")


def build_defense_b_dataset(
    train_df: pd.DataFrame,
    clean_predictor: TransformerRegressor,
    clean_model_hash: str,
    vocab: list[str],
    seed: int,
    output_csv: Path,
) -> None:
    """Build Defense B (Stress) training set with clean_model_teacher predictions."""
    rng = np.random.default_rng(seed)
    attacks = [
        SubstitutionAttack(rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1),
        InsertionAttack(rng, allowed_tokens=vocab, attack_budget=1),
        DeletionAttack(rng, attack_budget=1),
        RearrangementAttack(rng, window_size=3),
    ]

    rows: list[dict[str, Any]] = []
    candidates_to_predict: list[tuple[int, str, str, str, bool]] = []

    for idx, row in train_df.iterrows():
        source_idx = int(row.name if isinstance(row.name, int) else idx)
        orig_rep = str(row["original_representation"])
        target_val = float(row["property_value"])

        # Base clean row with measured label
        rows.append({
            "source_index": source_idx,
            "original_representation": orig_rep,
            "adversarial_representation": orig_rep,
            "attack_family": "clean",
            "canonical_equivalent": True,
            "label_basis": "measured",
            "target_value": target_val,
            "teacher_model_hash": None,
            "property_value": target_val,
        })

        # Generate chemistry-changing stress candidates
        tokens = tokenize(orig_rep)
        for attack in attacks:
            family = attack.metadata().get("attack_type", getattr(attack, "name", "unknown"))
            for outcome in attack.generate(tokens, n_variants=2):
                adv_rep = outcome.adversarial_representation
                is_eq = is_canonically_equivalent(orig_rep, adv_rep)
                candidates_to_predict.append((source_idx, orig_rep, adv_rep, family, is_eq))

    # Predict teacher labels for all stress candidates
    if candidates_to_predict:
        adv_reps = [c[2] for c in candidates_to_predict]
        teacher_preds = clean_predictor.predict(adv_reps)

        for (src_idx, orig_rep, adv_rep, family, is_eq), teacher_val in zip(candidates_to_predict, teacher_preds):
            rows.append({
                "source_index": src_idx,
                "original_representation": orig_rep,
                "adversarial_representation": adv_rep,
                "attack_family": family,
                "canonical_equivalent": is_eq,
                "label_basis": "clean_model_teacher",
                "target_value": float(teacher_val),
                "teacher_model_hash": clean_model_hash,
                "property_value": float(teacher_val),
            })

    aug_df = pd.DataFrame(rows)
    aug_df["original_representation"] = aug_df["adversarial_representation"]
    aug_df.to_csv(output_csv, index=False)
    print(f"Defense B dataset built: {len(aug_df)} rows saved to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Paired Robustness Benchmark")
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Evaluation split")
    parser.add_argument("--seed", type=int, default=20260815, help="Random seed")
    parser.add_argument("--n-variants", type=int, default=5, help="Variants per attack")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_id = args.run_id or f"run_{int(time.time())}"
    run_dir = repo_root / "results" / "phase2_paired_robustness" / run_id

    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")

    print(f"=== Starting Phase 2 Paired Robustness Run: {run_id} ===")
    run_dir.mkdir(parents=True)

    data_dir = repo_root / "data"
    models_dir = repo_root / "results" / "models"
    clean_model_dir = models_dir / "transformer_regressor"

    # Load dataset, splits, vocab, clean predictor
    df = pd.read_csv(data_dir / "processed" / "processed.csv")
    with (data_dir / "processed" / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    with (data_dir / "processed" / "vocab.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    clean_predictor = load_predictor(clean_model_dir, vocab)
    clean_model_hash = compute_file_hash(clean_model_dir / "model.pt")
    scaler_hash = compute_file_hash(clean_model_dir / "scaler.json")

    # Step 4: Build Defended Training Datasets & Train Defense A & B
    train_indices = splits["train"]
    train_df = df.iloc[train_indices].copy()

    def_a_dir = run_dir / "defense_a_model"
    def_b_dir = run_dir / "defense_b_model"

    def_a_csv = run_dir / "defense_a_train.csv"
    def_b_csv = run_dir / "defense_b_train.csv"

    print("Building Defense A (Randomization) training set...")
    build_defense_a_dataset(train_df, vocab, seed=args.seed, output_csv=def_a_csv)

    print("Building Defense B (Teacher-Labeled Stress) training set...")
    build_defense_b_dataset(train_df, clean_predictor, clean_model_hash, vocab, seed=args.seed, output_csv=def_b_csv)

    # Train Defense A using frozen baseline scaler
    print("Training Defense A (Randomization Defense)...")
    train_model(
        augmented_train_path=str(def_a_csv),
        out_dir=str(def_a_dir),
        scaler_path=str(clean_model_dir / "scaler.json"),
        write_back_config=False,
        seed=args.seed,
    )
    def_a_predictor = load_predictor(def_a_dir, vocab)

    # Train Defense B using frozen baseline scaler
    print("Training Defense B (Stress Defense)...")
    train_model(
        augmented_train_path=str(def_b_csv),
        out_dir=str(def_b_dir),
        scaler_path=str(clean_model_dir / "scaler.json"),
        write_back_config=False,
        seed=args.seed,
    )
    def_b_predictor = load_predictor(def_b_dir, vocab)

    # Step 1 & 2: Generate Frozen Model-Independent Candidate Bank
    eval_indices = splits[args.split]
    eval_df = df.iloc[eval_indices]
    eval_samples = list(zip(eval_df.index.tolist(), eval_df["polymer_id"].astype(str), eval_df["original_representation"].astype(str)))

    rng = np.random.default_rng(args.seed)
    attacks = [
        SubstitutionAttack(rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1),
        InsertionAttack(rng, allowed_tokens=vocab, attack_budget=1),
        DeletionAttack(rng, attack_budget=1),
        RearrangementAttack(rng, window_size=3),
        SmilesRandomizationAttack(rng, n_attempts=10),
    ]

    print(f"Generating frozen candidate bank on {args.split} set ({len(eval_samples)} polymers)...")
    candidates = generate_paired_candidate_bank(
        samples=eval_samples,
        attacks=attacks,
        n_variants=args.n_variants,
        seed=args.seed,
    )
    print(f"Candidate bank generated: {len(candidates)} candidates.")

    # Step 3 & 5: Score Clean Baseline, Defense A, and Defense B on SAME Candidate Bank
    print("Scoring Clean Baseline Model...")
    clean_records = score_model_on_candidate_bank(candidates, clean_predictor)

    print("Scoring Defense A (Randomization Defense) Model...")
    def_a_records = score_model_on_candidate_bank(candidates, def_a_predictor)

    print("Scoring Defense B (Stress Defense) Model...")
    def_b_records = score_model_on_candidate_bank(candidates, def_b_predictor)

    # Step 6 & 8: Paired Statistics & Transfer Matrices
    stats_a = compute_paired_statistics(clean_records, def_a_records, n_bootstrap=2000, seed=args.seed)
    stats_b = compute_paired_statistics(clean_records, def_b_records, n_bootstrap=2000, seed=args.seed)

    # Validity stratification summary per Step 2
    validity_summary: dict[str, dict[str, int]] = {}
    for c in candidates:
        fam = c.attack_family
        if fam not in validity_summary:
            validity_summary[fam] = {
                "proposed": 0, "syntactically_valid": 0, "rdkit_valid": 0,
                "canonical_equivalent": 0, "eligible": 0
            }
        validity_summary[fam]["proposed"] += 1
        if c.validity_status in ("valid", "implausible"):
            validity_summary[fam]["syntactically_valid"] += 1
        if c.validity_status == "valid":
            validity_summary[fam]["rdkit_valid"] += 1
        if c.canonical_equivalent:
            validity_summary[fam]["canonical_equivalent"] += 1
        if c.is_eligible:
            validity_summary[fam]["eligible"] += 1

    # Clean Performance MAE Table (Step 7)
    clean_perf = {
        "clean_baseline": json.load((clean_model_dir / "metrics.json").open()),
        "defense_a_randomization": json.load((def_a_dir / "metrics.json").open()),
        "defense_b_stress": json.load((def_b_dir / "metrics.json").open()),
    }
    clean_perf_rows = []
    for m_name, m_dict in clean_perf.items():
        clean_perf_rows.append({
            "model": m_name,
            "val_mae": m_dict.get("val_mae", m_dict.get("best_val_mae")),
            "test_mae": m_dict.get("test_mae"),
            "n_train": m_dict.get("n_train"),
        })
    clean_perf_df = pd.DataFrame(clean_perf_rows)
    clean_perf_df.to_csv(run_dir / "clean_performance_table.csv", index=False)

    # Transfer Matrix CSV (Step 8)
    transfer_rows = []
    all_families = sorted(list(validity_summary.keys()))
    for fam in all_families:
        c_rec_fam = [r for r in clean_records if r.attack_family == fam and r.is_eligible]
        a_rec_fam = [r for r in def_a_records if r.attack_family == fam and r.is_eligible]
        b_rec_fam = [r for r in def_b_records if r.attack_family == fam and r.is_eligible]

        c_drifts = [r.abs_drift for r in c_rec_fam]
        a_drifts = [r.abs_drift for r in a_rec_fam]
        b_drifts = [r.abs_drift for r in b_rec_fam]

        transfer_rows.append({
            "attack_family": fam,
            "eligible_N": len(c_drifts),
            "clean_mean_drift": float(np.mean(c_drifts)) if c_drifts else None,
            "clean_p95_drift": float(np.percentile(c_drifts, 95)) if c_drifts else None,
            "clean_max_drift": float(np.max(c_drifts)) if c_drifts else None,
            "def_a_mean_drift": float(np.mean(a_drifts)) if a_drifts else None,
            "def_a_p95_drift": float(np.percentile(a_drifts, 95)) if a_drifts else None,
            "def_a_max_drift": float(np.max(a_drifts)) if a_drifts else None,
            "def_b_mean_drift": float(np.mean(b_drifts)) if b_drifts else None,
            "def_b_p95_drift": float(np.percentile(b_drifts, 95)) if b_drifts else None,
            "def_b_max_drift": float(np.max(b_drifts)) if b_drifts else None,
        })
    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df.to_csv(run_dir / "robustness_matrix.csv", index=False)

    # Paired Improvement Matrix CSV (Step 8)
    improvement_rows = []
    for fam in all_families:
        st_a = stats_a.get(fam, {})
        st_b = stats_b.get(fam, {})
        improvement_rows.append({
            "attack_family": fam,
            "eligible_N": st_a.get("paired_eligible_N", 0),
            "def_a_mean_paired_delta": st_a.get("mean_paired_delta"),
            "def_a_ci_95": f"[{st_a.get('ci_95_lower', 0):.4f}, {st_a.get('ci_95_upper', 0):.4f}]",
            "def_a_fraction_improved": st_a.get("fraction_improved"),
            "def_b_mean_paired_delta": st_b.get("mean_paired_delta"),
            "def_b_ci_95": f"[{st_b.get('ci_95_lower', 0):.4f}, {st_b.get('ci_95_upper', 0):.4f}]",
            "def_b_fraction_improved": st_b.get("fraction_improved"),
        })
    improvement_df = pd.DataFrame(improvement_rows)
    improvement_df.to_csv(run_dir / "paired_improvement_matrix.csv", index=False)

    # Step 10: Persist JSON records and summaries
    with (run_dir / "candidate_bank.jsonl").open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c.to_dict(), sort_keys=True) + "\n")

    with (run_dir / "clean_records.jsonl").open("w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    with (run_dir / "defense_a_records.jsonl").open("w", encoding="utf-8") as f:
        for r in def_a_records:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    with (run_dir / "defense_b_records.jsonl").open("w", encoding="utf-8") as f:
        for r in def_b_records:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    summary_json = {
        "run_id": run_id,
        "evaluation_split": args.split,
        "validity_stratification": validity_summary,
        "paired_statistics_defense_a_randomization": stats_a,
        "paired_statistics_defense_b_stress": stats_b,
        "clean_performance": clean_perf,
    }
    with (run_dir / "robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary_json), f, indent=2, sort_keys=True)

    reproducibility = {
        "run_id": run_id,
        "git_commit": get_git_commit(repo_root),
        "split_name": args.split,
        "split_sample_count": len(eval_samples),
        "vocab_hash": compute_file_hash(data_dir / "processed" / "vocab.json"),
        "processed_data_hash": compute_file_hash(data_dir / "processed" / "processed.csv"),
        "scaler_hash": scaler_hash,
        "clean_model_hash": clean_model_hash,
        "defense_a_model_hash": compute_file_hash(def_a_dir / "model.pt"),
        "defense_b_model_hash": compute_file_hash(def_b_dir / "model.pt"),
        "seed": args.seed,
        "n_variants": args.n_variants,
    }
    with (run_dir / "reproducibility.json").open("w", encoding="utf-8") as f:
        json.dump(reproducibility, f, indent=2, sort_keys=True)

    print(f"\n=== Phase 2 Run Completed Successfully: {run_id} ===")
    print(f"Results persisted in: {run_dir}")


if __name__ == "__main__":
    main()
