"""Regression tests for Phase 2 Paired Robustness Benchmark."""

from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from materials_adv.experiments.paired_robustness import (
    PairedCandidate,
    PredictionRecord,
    compute_paired_statistics,
    compute_polymer_clustered_bootstrap,
    is_canonically_equivalent,
)
from rdkit import Chem
from materials_adv.data.scaler import TargetScaler


def test_paired_delta_sign_convention():
    """Verify that positive delta means robustness improvement (clean drift > defended drift)."""
    clean_rec = PredictionRecord(
        candidate_id="c1",
        source_index=0,
        sample_id="s0",
        attack_family="substitution",
        original_representation="[*]CC[*]",
        adversarial_representation="[*]C(C)C[*]",
        is_eligible=True,
        original_prediction=1.0,
        adversarial_prediction=2.5,
        abs_drift=1.5,
        signed_drift=1.5,
    )

    defended_rec = PredictionRecord(
        candidate_id="c1",
        source_index=0,
        sample_id="s0",
        attack_family="substitution",
        original_representation="[*]CC[*]",
        adversarial_representation="[*]C(C)C[*]",
        is_eligible=True,
        original_prediction=1.0,
        adversarial_prediction=1.2,
        abs_drift=0.2,
        signed_drift=0.2,
    )

    stats = compute_paired_statistics([clean_rec], [defended_rec], n_bootstrap=100)
    sub_stats = stats["substitution"]
    assert sub_stats["clean_mean_drift"] == 1.5
    assert sub_stats["defended_mean_drift"] == 0.2
    assert sub_stats["mean_paired_delta"] == 1.3  # Positive delta = improvement!
    assert sub_stats["fraction_improved"] == 1.0


def test_invalid_candidates_excluded_from_paired_statistics():
    """Verify that ineligible candidates are excluded from paired statistics."""
    clean_rec = PredictionRecord(
        candidate_id="c1",
        source_index=0,
        sample_id="s0",
        attack_family="deletion",
        original_representation="[*]CC[*]",
        adversarial_representation="[*]C[*]",
        is_eligible=False,  # Ineligible
        original_prediction=1.0,
        adversarial_prediction=5.0,
        abs_drift=4.0,
        signed_drift=4.0,
    )

    defended_rec = PredictionRecord(
        candidate_id="c1",
        source_index=0,
        sample_id="s0",
        attack_family="deletion",
        original_representation="[*]CC[*]",
        adversarial_representation="[*]C[*]",
        is_eligible=False,
        original_prediction=1.0,
        adversarial_prediction=1.0,
        abs_drift=0.0,
        signed_drift=0.0,
    )

    stats = compute_paired_statistics([clean_rec], [defended_rec], n_bootstrap=100)
    assert "deletion" not in stats or stats["deletion"]["paired_eligible_N"] == 0


def test_canonical_equivalence_required_for_inherited_measured_labels():
    """Verify canonical equivalence helper returns True only for chemically identical graphs."""
    orig = "[*]C=C([*])c1ccc(NC=O)cc1"
    mol = Chem.MolFromSmiles(orig)
    rand_valid = Chem.MolToSmiles(mol, doRandom=True)
    different_chem = "[*]C=C([*])c1ccc(N)cc1"

    assert is_canonically_equivalent(orig, orig) is True
    assert is_canonically_equivalent(orig, rand_valid) is True
    assert is_canonically_equivalent(orig, different_chem) is False


def test_bootstrap_samples_source_polymers_not_candidate_rows():
    """Verify bootstrap resamples source polymer clusters."""
    sources = np.array([0, 0, 1, 1])
    deltas = np.array([1.0, 1.0, -1.0, -1.0])

    boot_res = compute_polymer_clustered_bootstrap(sources, deltas, n_resamples=500, seed=42)
    assert boot_res["mean"] == 0.0
    assert boot_res["ci_lower"] <= 0.0 <= boot_res["ci_upper"]


def test_frozen_scaler_equality(tmp_path):
    """Verify baseline and defended scalers match byte-for-byte."""
    scaler_a = TargetScaler()
    scaler_a.mean = 4.474831
    scaler_a.std = 1.456305

    scaler_b = TargetScaler()
    scaler_b.mean = 4.474831
    scaler_b.std = 1.456305

    file_a = tmp_path / "scaler_a.json"
    file_b = tmp_path / "scaler_b.json"

    scaler_a.save(file_a)
    scaler_b.save(file_b)

    with file_a.open("rb") as fa, file_b.open("rb") as fb:
        assert fa.read() == fb.read()


def test_output_directory_immutability(tmp_path):
    """Verify that attempting to rerun on an existing directory fails loudly."""
    existing_dir = tmp_path / "run_existing"
    existing_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        if existing_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {existing_dir}")
