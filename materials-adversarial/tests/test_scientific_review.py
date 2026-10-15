from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from materials_adv.evaluation.multiview import (
    canonical_smiles,
    equivalent_views,
    evaluate_multiview,
    molecule_bootstrap_mae_delta,
    trimmed_mean,
)
from materials_adv.oracle.candidate_selection import select_oracle_candidates
from materials_adv.data.splits import grouped_split
from materials_adv.data.graph_dataset import GraphDataset


def test_equivalent_views_are_deterministic_unique_and_graph_equivalent():
    source = "[*]CCOc1ccccc1[*]"
    a = equivalent_views(source, 12, 17)
    b = equivalent_views(source, 12, 17)
    assert a == b
    assert len(a) == len(set(a))
    assert all(canonical_smiles(value) == canonical_smiles(source) for value in a)


def test_symmetric_molecules_report_actual_unique_views_without_duplication():
    views = equivalent_views("[*]CC[*]", 100, 2)
    assert len(views) < 100
    assert len(views) == len(set(views))


def test_multiview_aggregates_per_molecule_not_per_view():
    frame = pd.DataFrame({
        "polymer_id": ["a", "b"],
        "original_representation": ["[*]CCOc1ccccc1[*]", "[*]CCNCC[*]"],
        "property_value": [1.0, 2.0],
    })
    predictor = lambda values: np.asarray([len(v) / 10 for v in values])
    result = evaluate_multiview(frame, predictor, n_views=8, seed=3)
    assert len(result) == 2
    assert (result["unique_views"] <= 8).all()
    assert np.isfinite(result["prediction_std_eV"]).all()
    assert "source_prediction_eV" in result


def test_trimmed_mean_and_molecule_bootstrap_do_not_zero_fill():
    assert trimmed_mean(np.array([0.0, 1.0, 2.0, 100.0]), 0.25) == 1.5
    frame = pd.DataFrame({
        "target_eV": [0.0, 1.0, 2.0],
        "canonical_prediction_eV": [1.0, 2.0, 3.0],
        "mean_prediction_eV": [0.0, 1.0, 2.0],
    })
    result = molecule_bootstrap_mae_delta(frame, n_resamples=100, seed=1)
    assert result["unit"] == "molecule"
    assert result["mean_mae_delta_eV"] == -1.0


def test_oracle_shortlist_is_valid_deduplicated_and_source_balanced():
    rows = pd.DataFrame([
        {"source_smiles": "CC", "candidate_smiles": "CO", "attack_family": "a", "model_drift_eV": 3.0, "true_label": 9.0},
        {"source_smiles": "CC", "candidate_smiles": "OC", "attack_family": "b", "model_drift_eV": 2.0, "true_label": 9.0},
        {"source_smiles": "CC", "candidate_smiles": "CN", "attack_family": "b", "model_drift_eV": 1.0, "true_label": 9.0},
        {"source_smiles": "CCC", "candidate_smiles": "CCO", "attack_family": "a", "model_drift_eV": 0.5, "true_label": 8.0},
        {"source_smiles": "CCC", "candidate_smiles": "broken", "attack_family": "a", "model_drift_eV": 10.0, "true_label": 8.0},
    ])
    result = select_oracle_candidates(rows, n=4, max_per_source=1)
    assert len(result) == 2
    assert result["candidate_canonical"].is_unique
    assert result["source_canonical"].value_counts().max() == 1
    assert "true_label" not in result


def test_oracle_shortlist_requires_scientific_metadata():
    with pytest.raises(ValueError):
        select_oracle_candidates(pd.DataFrame({"source_smiles": ["CC"]}))


def test_grouped_split_seed_is_reproducible_and_effective():
    keys = [f"group-{i}" for i in range(30)]
    assert grouped_split(keys, seed=7) == grouped_split(keys, seed=7)
    assert grouped_split(keys, seed=7) != grouped_split(keys, seed=8)


def test_graph_dataset_fails_closed_on_invalid_candidate():
    frame = pd.DataFrame({"original_representation": ["not-smiles"], "property_value": [1.0]})
    with pytest.raises(ValueError, match="strict=True"):
        GraphDataset(frame)[0]


def test_chemistry_changing_bank_schema_has_no_candidate_label():
    source = Path("scripts/generate_phase3_banks.py").read_text()
    deletion = Path("scripts/generate_deletion_candidates.py").read_text()
    assert '"candidate_target": None' in source
    assert '"candidate_target": None' in deletion
    assert "candidate_target_requires_independent_physical_oracle" in source
    assert "candidate_target_requires_independent_physical_oracle" in deletion
