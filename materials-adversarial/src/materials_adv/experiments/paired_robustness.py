"""Core experiment engine for Phase 2: Controlled Paired Adversarial Baseline & Defense Benchmark.

Implements model-independent candidate bank generation, validity stratification,
canonical equivalence verification, teacher-labeled dataset augmentation,
paired robustness scoring, and polymer-clustered bootstrap statistics.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import torch
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from rdkit import Chem

from ..attacks.base import BaseAttack
from ..attacks.deletion import DeletionAttack
from ..attacks.insertion import InsertionAttack
from ..attacks.randomization import SmilesRandomizationAttack
from ..attacks.rearrangement import RearrangementAttack
from ..attacks.substitution import SubstitutionAttack
from ..data.scaler import TargetScaler
from ..data.tokenizer import tokenize
from ..evaluation.records import make_attack_id
from ..models.regression import TransformerRegressor
from ..models.transformer import TransformerRegressorModel
from ..training.train import train as train_model
from ..utils.io import to_jsonable
from ..validation.pipeline import validate


def canonicalize_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or None if unparseable."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_canonically_equivalent(orig_smiles: str, pert_smiles: str) -> bool:
    """Return True if both representations yield identical RDKit canonical SMILES."""
    c_orig = canonicalize_smiles(orig_smiles)
    c_pert = canonicalize_smiles(pert_smiles)
    return c_orig is not None and c_orig == c_pert


@dataclass(frozen=True, slots=True)
class PairedCandidate:
    candidate_id: str
    source_index: int
    sample_id: str
    original_representation: str
    adversarial_representation: str
    attack_family: str
    attack_budget: int
    number_of_changes: int
    edited_positions: tuple[int, ...]
    validity_status: str
    plausibility_status: str
    canonical_equivalent: bool
    is_eligible: bool
    attack_params: dict[str, Any]
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["edited_positions"] = list(self.edited_positions)
        return to_jsonable(data)


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    candidate_id: str
    source_index: int
    sample_id: str
    attack_family: str
    original_representation: str
    adversarial_representation: str
    is_eligible: bool
    original_prediction: float
    adversarial_prediction: float
    abs_drift: float
    signed_drift: float

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def generate_paired_candidate_bank(
    samples: Sequence[tuple[int, str, str]],  # (source_index, sample_id, original_representation)
    attacks: Sequence[BaseAttack],
    *,
    n_variants: int = 5,
    seed: int | None = 20260815,
) -> list[PairedCandidate]:
    """Generate a single model-independent candidate bank for validation or test set."""
    candidates: list[PairedCandidate] = []

    for source_idx, sample_id, orig_rep in samples:
        tokens = tokenize(orig_rep)
        for attack in attacks:
            metadata = attack.metadata()
            attack_family = metadata.get("attack_type", getattr(attack, "name", "unknown"))

            for ordinal, outcome in enumerate(attack.generate(tokens, n_variants=n_variants)):
                cand_id = make_attack_id(sample_id, attack_family, seed, ordinal)
                adv_rep = outcome.adversarial_representation

                # Validation & canonical equivalence checks
                val_res = validate(adv_rep, check_plausibility=True)
                val_status = val_res.status.value
                plaus_status = (
                    "plausible"
                    if val_status == "valid"
                    else "implausible" if val_status == "implausible" else "unchecked"
                )

                canon_eq = is_canonically_equivalent(orig_rep, adv_rep)

                # Eligibility classification per Step 2
                if attack_family == "randomization":
                    is_eligible = (val_status == "valid") and canon_eq
                else:
                    is_eligible = (val_status == "valid")

                candidates.append(
                    PairedCandidate(
                        candidate_id=cand_id,
                        source_index=source_idx,
                        sample_id=sample_id,
                        original_representation=orig_rep,
                        adversarial_representation=adv_rep,
                        attack_family=attack_family,
                        attack_budget=int(metadata.get("attack_budget", 1)),
                        number_of_changes=outcome.number_of_changes,
                        edited_positions=outcome.edit_positions,
                        validity_status=val_status,
                        plausibility_status=plaus_status,
                        canonical_equivalent=canon_eq,
                        is_eligible=is_eligible,
                        attack_params={**metadata, **outcome.params},
                        seed=seed,
                    )
                )

    cand_ids = [c.candidate_id for c in candidates]
    if len(cand_ids) != len(set(cand_ids)):
        raise ValueError("Candidate IDs are not unique; duplicate attack configurations detected.")

    return candidates


def score_model_on_candidate_bank(
    candidates: Sequence[PairedCandidate],
    predictor: TransformerRegressor,
) -> list[PredictionRecord]:
    """Score a model against the frozen candidate bank."""
    if not candidates:
        return []

    unique_reps = sorted(
        {
            rep
            for cand in candidates
            for rep in (cand.original_representation, cand.adversarial_representation)
        }
    )
    preds = predictor.predict(unique_reps)
    pred_map = {rep: float(p) for rep, p in zip(unique_reps, preds)}

    records: list[PredictionRecord] = []
    for cand in candidates:
        orig_p = pred_map[cand.original_representation]
        adv_p = pred_map[cand.adversarial_representation]
        signed_drift = adv_p - orig_p
        abs_drift = abs(signed_drift)

        records.append(
            PredictionRecord(
                candidate_id=cand.candidate_id,
                source_index=cand.source_index,
                sample_id=cand.sample_id,
                attack_family=cand.attack_family,
                original_representation=cand.original_representation,
                adversarial_representation=cand.adversarial_representation,
                is_eligible=cand.is_eligible,
                original_prediction=orig_p,
                adversarial_prediction=adv_p,
                abs_drift=abs_drift,
                signed_drift=signed_drift,
            )
        )

    return records


def compute_polymer_clustered_bootstrap(
    source_indices: np.ndarray,
    deltas: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 20260815,
) -> dict[str, float]:
    """Compute bootstrap CI for paired robustness deltas clustered by source polymer."""
    if len(deltas) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    unique_polymers = np.unique(source_indices)
    n_polymers = len(unique_polymers)
    rng = np.random.default_rng(seed)

    # Group delta arrays by source polymer
    polymer_deltas = [deltas[source_indices == p] for p in unique_polymers]

    boot_means = np.zeros(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled_polymer_idx = rng.choice(n_polymers, size=n_polymers, replace=True)
        sample_deltas = np.concatenate([polymer_deltas[idx] for idx in sampled_polymer_idx])
        boot_means[i] = np.mean(sample_deltas)

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return {
        "mean": float(np.mean(deltas)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compute_paired_statistics(
    clean_records: Sequence[PredictionRecord],
    defended_records: Sequence[PredictionRecord],
    n_bootstrap: int = 2000,
    seed: int = 20260815,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compute paired robustness deltas and polymer-clustered CIs.
    
    Formula: paired_delta = clean_drift - defended_drift
    Positive delta = robustness improvement (defended model has lower drift).
    """
    clean_map = {r.candidate_id: r for r in clean_records}
    defended_map = {r.candidate_id: r for r in defended_records}

    if set(clean_map.keys()) != set(defended_map.keys()):
        raise ValueError("Clean and defended record candidate IDs do not match!")

    by_family: dict[str, list[tuple[int, float, float, float]]] = {}

    for cand_id, clean_rec in clean_map.items():
        if not clean_rec.is_eligible:
            continue

        defended_rec = defended_map[cand_id]
        if clean_rec.original_representation != defended_rec.original_representation or \
           clean_rec.adversarial_representation != defended_rec.adversarial_representation:
            raise ValueError(f"Candidate representation mismatch for candidate {cand_id!r}")

        clean_drift = clean_rec.abs_drift
        defended_drift = defended_rec.abs_drift
        delta = clean_drift - defended_drift  # Positive = improvement

        by_family.setdefault(clean_rec.attack_family, []).append(
            (clean_rec.source_index, clean_drift, defended_drift, delta)
        )

    results: dict[str, Any] = {}
    for family in sorted(by_family.keys()):
        data = by_family[family]
        sources = np.array([d[0] for d in data], dtype=int)
        clean_drifts = np.array([d[1] for d in data], dtype=float)
        defended_drifts = np.array([d[2] for d in data], dtype=float)
        deltas = np.array([d[3] for d in data], dtype=float)

        n_eligible = len(deltas)
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))

        # Improvements, worsenings, unchanged counts
        improved_mask = deltas > tolerance
        worsened_mask = deltas < -tolerance
        unchanged_mask = np.abs(deltas) <= tolerance

        frac_improved = float(np.mean(improved_mask))
        frac_worsened = float(np.mean(worsened_mask))
        frac_unchanged = float(np.mean(unchanged_mask))

        boot_stats = compute_polymer_clustered_bootstrap(
            sources, deltas, n_resamples=n_bootstrap, seed=seed
        )

        results[family] = {
            "paired_eligible_N": n_eligible,
            "mean_paired_delta": mean_delta,
            "median_paired_delta": median_delta,
            "ci_95_lower": boot_stats["ci_lower"],
            "ci_95_upper": boot_stats["ci_upper"],
            "fraction_improved": frac_improved,
            "fraction_worsened": frac_worsened,
            "fraction_unchanged": frac_unchanged,
            "clean_mean_drift": float(np.mean(clean_drifts)),
            "defended_mean_drift": float(np.mean(defended_drifts)),
            "clean_p95_drift": float(np.percentile(clean_drifts, 95)),
            "defended_p95_drift": float(np.percentile(defended_drifts, 95)),
            "clean_max_drift": float(np.max(clean_drifts)),
            "defended_max_drift": float(np.max(defended_drifts)),
        }

    return results
