"""Structural tests for the reusable, non-overwriting experiment workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.evaluation.attack_metrics import SuccessCriterion
from materials_adv.experiments.pipeline import (
    AttackCandidate,
    ExperimentPipeline,
    compare_paired_records,
    score_candidate_bank,
)


def test_candidate_metadata_serializes_numpy_scalars() -> None:
    candidate = AttackCandidate(
        candidate_id="c", sample_id="s", original_representation="CC",
        adversarial_representation="CN", attack_type="insertion", attack_budget=1,
        number_of_changes=1, edited_positions=(np.int64(1),),
        attack_params={"position": np.int64(1), "probability": np.float64(0.5)}, seed=1,
    )
    payload = candidate.to_dict()
    assert payload["edited_positions"] == [1]
    assert payload["attack_params"] == {"position": 1, "probability": 0.5}
    json.dumps(payload)


class LengthPredictor:
    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.target_units = "synthetic"

    def predict(self, psmiles) -> np.ndarray:
        return np.asarray([len(value) * self.scale for value in psmiles], dtype=float)


def test_full_workflow_reuses_one_candidate_bank_and_never_overwrites(tmp_path) -> None:
    run_dir = tmp_path / "run-001"

    def train_clean(model_dir: Path) -> LengthPredictor:
        model_dir.mkdir()
        (model_dir / "marker.txt").write_text("clean")
        return LengthPredictor(1.0)

    def build_attacks(_predictor):
        return [
            SubstitutionAttack(
                np.random.default_rng(7),
                allowed_tokens=["C", "N", "O"],
            )
        ]

    def augment(records, output_path: Path) -> Path:
        output_path.write_text("n_records\n" + str(len(records)) + "\n")
        return output_path

    def train_defended(dataset_path: Path, model_dir: Path) -> LengthPredictor:
        assert dataset_path.exists()
        model_dir.mkdir()
        (model_dir / "marker.txt").write_text("defended")
        return LengthPredictor(0.5)

    result = ExperimentPipeline(run_dir).run(
        samples=[("sample-1", "CCO")],
        train_clean=train_clean,
        build_attacks=build_attacks,
        criterion=SuccessCriterion(min_abs_drift=0.0, require_valid=False),
        n_variants=2,
        seed=7,
        augment=augment,
        train_defended=train_defended,
    )

    clean = [json.loads(line) for line in result.clean_records_path.read_text().splitlines()]
    defended = [
        json.loads(line)
        for line in result.defended_records_path.read_text().splitlines()
    ]
    assert clean
    assert [row["attack_id"] for row in clean] == [
        row["attack_id"] for row in defended
    ]
    assert [row["adversarial_representation"] for row in clean] == [
        row["adversarial_representation"] for row in defended
    ]
    comparison = json.loads(result.comparison_path.read_text())
    assert comparison["paired"] is True
    assert comparison["n_candidates"] == len(clean)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ExperimentPipeline(run_dir).run(
            samples=[],
            train_clean=train_clean,
            build_attacks=build_attacks,
            criterion=SuccessCriterion(),
            n_variants=1,
            seed=7,
        )


def test_paired_comparison_rejects_different_candidate_order() -> None:
    from materials_adv.evaluation.records import AttackRecord

    def record(attack_id: str) -> AttackRecord:
        return AttackRecord(
            attack_id=attack_id,
            sample_id="sample",
            original_representation="CC",
            adversarial_representation="CN",
            attack_type="substitution",
            attack_budget=1,
            number_of_changes=1,
            edited_positions=(1,),
            validity_status="valid",
            plausibility_status="plausible",
            signed_prediction_drift=1.0,
            absolute_prediction_drift=1.0,
        )

    with pytest.raises(ValueError, match="identical ordered candidate IDs"):
        compare_paired_records(
            [record("a"), record("b")],
            [record("b"), record("a")],
            criterion=SuccessCriterion(),
        )


def test_scoring_rejects_predictor_cardinality_mismatch() -> None:
    class BrokenPredictor:
        target_units = "synthetic"

        def predict(self, _psmiles):
            return np.asarray([], dtype=float)

    candidate = AttackCandidate(
        candidate_id="sample:substitution:1:0",
        sample_id="sample",
        original_representation="CC",
        adversarial_representation="CN",
        attack_type="substitution",
        attack_budget=1,
        number_of_changes=1,
        edited_positions=(1,),
        attack_params={},
        seed=1,
    )
    with pytest.raises(ValueError, match="different number of predictions"):
        score_candidate_bank(
            [candidate],
            BrokenPredictor(),
            criterion=SuccessCriterion(),
        )
