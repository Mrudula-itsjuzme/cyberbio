#!/usr/bin/env python3
"""Generate Phase 3 Frozen Candidate Banks (Validation Split).

Creates immutable randomization and substitution banks for testing
model robustness and branch specialization on unseen candidates.
Ensures no exact leakage from the Phase 2 training pairs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from materials_adv.domain.chemistry.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.substitution import SubstitutionAttack
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


def regenerate_phase2_training_candidates(
    train_reps: list[str], vocab: list[str], seed: int = 20260815
) -> set[str]:
    """Reconstruct Phase 2 candidates to filter out evaluation leakage."""
    rng = np.random.default_rng(seed)
    rand_attack = SmilesRandomizationAttack(rng, n_attempts=10)
    sub_attack = SubstitutionAttack(
        rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1
    )

    training_candidates = set()

    for rep in train_reps:
        tokens = tokenize(rep)
        r_outs = rand_attack.generate(tokens, n_variants=1)
        if r_outs:
            candidate_rand = r_outs[0].adversarial_representation
            if validate(candidate_rand).status.value == "valid" and candidate_rand != rep:
                training_candidates.add(candidate_rand)

        s_outs = sub_attack.generate(tokens, n_variants=1)
        if s_outs:
            candidate_sub = s_outs[0].adversarial_representation
            if validate(candidate_sub).status.value == "valid" and candidate_sub != rep:
                training_candidates.add(candidate_sub)

    return training_candidates


def generate_banks(
    output_dir: Path,
    seed: int = 20261101,  # different from phase 2 seed
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "processed"

    # 1. Load data
    df = pd.read_csv(data_dir / "processed.csv")
    with (data_dir / "splits.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    with (data_dir / "vocab.json").open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    train_df = df.iloc[splits["train"]]
    val_df = df.iloc[splits["val"]]

    # 2. Get Phase 2 candidates for exclusion
    print("Reconstructing Phase 2 training candidates for leakage audit...")
    phase2_train_reps = train_df["original_representation"].tolist()
    phase2_candidates = regenerate_phase2_training_candidates(phase2_train_reps, vocab)
    print(f"  Found {len(phase2_candidates)} unique candidates from Phase 2 training.")

    # 3. Generate Evaluation Banks
    rng = np.random.default_rng(seed)
    # We will generate up to 5 randomizations and 5 substitutions per val polymer,
    # then filter for validity and overlap.
    rand_attack = SmilesRandomizationAttack(rng, n_attempts=20)
    sub_attack = SubstitutionAttack(
        rng, allowed_tokens=vocab, role_preserving=True, attack_budget=1
    )

    val_reps = val_df["original_representation"].tolist()
    val_targets = val_df["property_value"].tolist()

    rand_bank: list[dict[str, Any]] = []
    sub_bank: list[dict[str, Any]] = []

    counts = {
        "rand_generated": 0,
        "rand_valid": 0,
        "rand_overlap": 0,
        "rand_final": 0,
        "sub_generated": 0,
        "sub_valid": 0,
        "sub_overlap": 0,
        "sub_final": 0,
    }

    print("Generating Validation Candidate Banks...")
    for i, (rep, target) in enumerate(zip(val_reps, val_targets)):
        source_id = f"val_{i:04d}"
        tokens = tokenize(rep)
        c_orig = get_canonical(rep)

        # A. Randomization
        r_outs = rand_attack.generate(tokens, n_variants=5)
        for j, out in enumerate(r_outs):
            counts["rand_generated"] += 1
            cand = out.adversarial_representation
            c_cand = get_canonical(cand)
            is_valid = validate(cand).status.value == "valid"
            is_equiv = (c_cand == c_orig) and (c_cand != "")

            if not is_valid or not is_equiv or cand == rep:
                continue
            counts["rand_valid"] += 1

            if cand in phase2_candidates:
                counts["rand_overlap"] += 1
                continue
            counts["rand_final"] += 1

            rand_bank.append({
                "source_id": source_id,
                "candidate_id": f"{source_id}_rand_{j}",
                "original_representation": rep,
                "candidate_representation": cand,
                "canonical_original": c_orig,
                "canonical_candidate": c_cand,
                "target": target,
                "generation_seed": seed,
                "valid": is_valid,
                "canonical_equivalent": is_equiv,
                "exclusion_reason": None,
            })

        # B. Substitution
        s_outs = sub_attack.generate(tokens, n_variants=5)
        for j, out in enumerate(s_outs):
            counts["sub_generated"] += 1
            cand = out.adversarial_representation
            c_cand = get_canonical(cand)
            is_valid = validate(cand).status.value == "valid"

            if not is_valid or cand == rep:
                continue
            counts["sub_valid"] += 1

            if cand in phase2_candidates:
                counts["sub_overlap"] += 1
                continue
            counts["sub_final"] += 1

            sub_bank.append({
                "source_id": source_id,
                "candidate_id": f"{source_id}_sub_{j}",
                "original_representation": rep,
                "candidate_representation": cand,
                "canonical_original": c_orig,
                "canonical_candidate": c_cand,
                "target": target,
                "generation_seed": seed,
                "valid": is_valid,
                "canonical_equivalent": (c_cand == c_orig) and (c_cand != ""),
                "exclusion_reason": None,
            })

    # 4. Save and Hash
    output_dir.mkdir(parents=True, exist_ok=True)
    rand_path = output_dir / "randomization_candidates.jsonl"
    sub_path = output_dir / "substitution_candidates.jsonl"
    manifest_path = output_dir / "manifest.json"
    reproducibility_path = output_dir / "reproducibility.json"

    with rand_path.open("w", encoding="utf-8") as f:
        for record in rand_bank:
            f.write(json.dumps(record) + "\n")

    with sub_path.open("w", encoding="utf-8") as f:
        for record in sub_bank:
            f.write(json.dumps(record) + "\n")

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    manifest = {
        "randomization_candidates.jsonl": sha256(rand_path),
        "substitution_candidates.jsonl": sha256(sub_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with reproducibility_path.open("w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "counts": counts,
            "leakage_audit": {
                "phase2_candidates_checked": len(phase2_candidates),
                "rand_overlaps_removed": counts["rand_overlap"],
                "sub_overlaps_removed": counts["sub_overlap"],
                "note": "Reconstructed Phase 2 pairs due to missing on-disk artifacts."
            }
        }, f, indent=2)

    print("\nPhase 3 Candidate Banks Generated Successfully.")
    print(f"Output Directory: {output_dir}")
    print(json.dumps(counts, indent=2))
    print(f"Manifest: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 Frozen Candidate Banks")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID for output dir")
    parser.add_argument("--seed", type=int, default=20261101, help="Seed for generation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "results" / "candidate_banks" / f"two_branch_validation_{args.run_id}"

    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists. Banks must be immutable.")

    generate_banks(output_dir, args.seed)


if __name__ == "__main__":
    main()
