#!/usr/bin/env python3
"""Run Phase 2 Controlled Paired Adversarial Baseline & Defense Benchmark.

Evaluates clean baseline and defended Transformer models on identical,
model-independent attack candidate banks for polyVERSE Bandgap (eV).
Generates a complete run manifest (reproducibility.json) and paired metric table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.evaluation.attack_metrics import SuccessCriterion
from materials_adv.experiments.pipeline import (
    generate_candidate_bank,
    score_candidate_bank,
    compare_paired_records,
    _write_jsonl_exclusive,
    _write_json_exclusive,
)
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.io import to_jsonable


def compute_file_hash(path: Path) -> str:
    """Return SHA256 hex digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_predictor(model_dir: Path, vocab: list[str], target_units: str = "eV") -> TransformerRegressor:
    """Load a TransformerRegressor checkpoint from directory."""
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
    state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    scaler = TargetScaler.load(model_dir / "scaler.json")
    return TransformerRegressor(model=model, vocab=vocab, scaler=scaler, target_units=target_units)


def get_git_commit(repo_root: Path) -> str:
    """Retrieve git HEAD commit hash if available."""
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


def run_paired_benchmark(
    *,
    output_dir: Path,
    split_name: str = "test",
    n_variants: int = 5,
    seed: int = 20260815,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    models_dir = repo_root / "results" / "models"

    clean_dir = models_dir / "transformer_regressor"
    defended_dir = models_dir / "transformer_defended"

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    # Load dataset & split
    df = pd.read_csv(data_dir / "processed" / "processed.csv")
    with (data_dir / "processed" / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)

    if split_name not in splits:
        raise ValueError(f"Unknown split {split_name!r}. Available: {list(splits.keys())}")

    split_indices = splits[split_name]
    split_df = df.iloc[split_indices]
    samples = list(zip(split_df["polymer_id"].astype(str), split_df["original_representation"].astype(str)))

    # Load vocabulary & predictors
    with (data_dir / "processed" / "vocab.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    clean_predictor = load_predictor(clean_dir, vocab)
    defended_predictor = load_predictor(defended_dir, vocab)

    # Initialize attacks
    rng = np.random.default_rng(seed)
    attacks = [
        SubstitutionAttack(rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1),
        InsertionAttack(rng, allowed_tokens=vocab, attack_budget=1),
        DeletionAttack(rng, attack_budget=1),
        RearrangementAttack(rng, window_size=3),
        SmilesRandomizationAttack(rng, n_attempts=10),
    ]

    # Generate shared candidate bank
    candidates = generate_candidate_bank(
        samples=samples,
        attacks=attacks,
        n_variants=n_variants,
        seed=seed,
    )

    # Score clean model
    criterion = SuccessCriterion(min_abs_drift=0.4619162976741791, require_valid=True)
    clean_records = score_candidate_bank(candidates, clean_predictor, criterion=criterion)

    # Score defended model on EXACT SAME candidate bank
    defended_records = score_candidate_bank(candidates, defended_predictor, criterion=criterion)

    # Perform paired comparison
    comparison = compare_paired_records(clean_records, defended_records, criterion=criterion)

    # Build detailed summary table
    summary_data = {
        "status": "completed",
        "split": split_name,
        "sample_count": len(samples),
        "total_candidates": len(candidates),
        "criterion": {
            "min_abs_drift_eV": criterion.min_abs_drift,
            "require_valid": criterion.require_valid,
        },
        "paired_metrics": comparison["defended_minus_clean"],
        "clean_summary": comparison["clean"],
        "defended_summary": comparison["defended"],
    }

    # Build reproducibility manifest
    manifest = {
        "git_commit": get_git_commit(repo_root),
        "split_name": split_name,
        "split_sample_count": len(samples),
        "vocab_size": len(vocab),
        "vocab_hash": compute_file_hash(data_dir / "processed" / "vocab.json"),
        "processed_data_hash": compute_file_hash(data_dir / "processed" / "processed.csv"),
        "clean_model": {
            "dir": str(clean_dir.relative_to(repo_root)),
            "model_hash": compute_file_hash(clean_dir / "model.pt"),
            "scaler_hash": compute_file_hash(clean_dir / "scaler.json"),
        },
        "defended_model": {
            "dir": str(defended_dir.relative_to(repo_root)),
            "model_hash": compute_file_hash(defended_dir / "model.pt"),
            "scaler_hash": compute_file_hash(defended_dir / "scaler.json"),
        },
        "seed": seed,
        "n_variants": n_variants,
    }

    # Write output files exclusively
    output_dir.mkdir(parents=True)
    _write_jsonl_exclusive(output_dir / "attack_candidates.jsonl", [c.to_dict() for c in candidates])
    _write_jsonl_exclusive(output_dir / "clean_attack_records.jsonl", [r.to_dict() for r in clean_records])
    _write_jsonl_exclusive(output_dir / "defended_attack_records.jsonl", [r.to_dict() for r in defended_records])
    _write_json_exclusive(output_dir / "paired_robustness.json", comparison)
    _write_json_exclusive(output_dir / "summary.json", summary_data)
    _write_json_exclusive(output_dir / "reproducibility.json", manifest)

    print(f"Paired benchmark completed successfully. Results saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 Paired Benchmark")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--n-variants", type=int, default=5, help="Number of variants per attack")
    parser.add_argument("--seed", type=int, default=20260815, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/phase2_paired_benchmark", help="Output directory")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / args.output_dir

    run_paired_benchmark(
        output_dir=output_path,
        split_name=args.split,
        n_variants=args.n_variants,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
