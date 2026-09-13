#!/usr/bin/env python3
"""Generate Phase 5 Deletion Candidate Bank (Validation Split).

Creates an immutable deletion bank for testing transfer robustness.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.data.tokenizer import tokenize
from materials_adv.validation.pipeline import validate


def get_canonical(smiles: str) -> str:
    """Return RDKit canonical SMILES or empty string if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def generate_bank(
    output_dir: Path,
    seed: int = 20261105,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"

    # 1. Load data
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)

    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]

    # 2. Extract Phase 4 Training set reps
    print("Collecting Phase 4 training representations for leakage audit...")
    # Phase 4 models were trained on original train reps + their randomizations and substitutions.
    # We will just verify that the *deleted* validation string does not exactly match any
    # original training string.
    train_reps = set(train_df["original_representation"].tolist())
    print(f"  Found {len(train_reps)} unique original training representations.")

    # 3. Generate Evaluation Bank
    rng = np.random.default_rng(seed)
    # We want exactly 1 token deleted.
    del_attack = DeletionAttack(
        rng, 
        attack_budget=1,
        protect_attachments=True,
        protect_ring_closures=True,
        protect_branches=True
    )

    val_reps = val_df["original_representation"].tolist()
    val_targets = val_df["property_value"].tolist()

    del_bank: list[dict[str, Any]] = []

    counts = {
        "generated": 0,
        "valid": 0,
        "overlap_with_clean_train": 0,
        "final": 0,
    }

    print("Generating Deletion Validation Bank...")
    for i, (rep, target) in enumerate(zip(val_reps, val_targets)):
        source_id = f"val_{i:04d}"
        tokens = tokenize(rep)
        c_orig = get_canonical(rep)

        # Generate up to 5 deletion attempts to find 1 valid one
        d_outs = del_attack.generate(tokens, n_variants=5)
        for j, out in enumerate(d_outs):
            counts["generated"] += 1
            cand = out.adversarial_representation
            c_cand = get_canonical(cand)
            is_valid = validate(cand).status.value == "valid"

            if not is_valid:
                continue
            counts["valid"] += 1

            if cand in train_reps:
                counts["overlap_with_clean_train"] += 1
                continue
                
            # MUST be chemistry-changing (canonical SMILES differs)
            if c_cand == c_orig:
                continue
                
            counts["final"] += 1

            del_bank.append({
                "source_id": source_id,
                "candidate_id": f"{source_id}_del_{j}",
                "original_representation": rep,
                "candidate_representation": cand,
                "canonical_original": c_orig,
                "canonical_candidate": c_cand,
                "target": target,
                "generation_seed": seed,
                "valid": is_valid,
            })
            # Take only the first valid, non-overlapping candidate for each source
            break

    print("\nGeneration Complete:")
    print("--------------------")
    print(f"Total Validation Sources: {len(val_reps)}")
    print(f"Generated Deletion Candidates: {counts['generated']}")
    print(f"Valid Deletion Candidates: {counts['valid']}")
    print(f"Overlap with Clean Train: {counts['overlap_with_clean_train']}")
    print(f"Final Deletion Candidates: {counts['final']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    bank_path = output_dir / "deletion_candidates.jsonl"
    with bank_path.open("w", encoding="utf-8") as f:
        for rec in del_bank:
            f.write(json.dumps(rec) + "\n")
            
    # Manifest
    def sha256(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()
        
    manifest = {
        "files": {
            "deletion_candidates.jsonl": sha256(bank_path),
        },
        "counts": counts,
        "seed": seed,
        "attack_budget": 1
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nSaved to: {output_dir}")
    print(f"Bank SHA256: {manifest['files']['deletion_candidates.jsonl']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/candidate_banks/deletion_transfer_phase5")
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    generate_bank(out_path)
