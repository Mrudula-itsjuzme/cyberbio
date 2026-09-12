"""Phase 4 scientific-contract and artifact tests without model training."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from materials_adv.attacks.base import AttackOutcome, BaseAttack
from materials_adv.evaluation.attack_metrics import SuccessCriterion
from materials_adv.evaluation.records import AttackRecord
from materials_adv.experiments.closed_loop import (
    ATTACK_TAXONOMY,
    AttackCategory,
    ClosedLoopExperiment,
    ClosedLoopSettings,
    DefenseSpec,
    paired_defense_delta,
    summarize_attack_records,
)
from materials_adv.validation.representation import canonical_graph_equivalent


class ScalePredictor:
    target_units = "synthetic"

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def predict(self, representations) -> np.ndarray:
        return np.asarray(
            [sum(ord(char) for char in value) * self.scale for value in representations],
            dtype=float,
        )


class NamedAttack(BaseAttack):
    def __init__(self, rng, name: str, variant_token: str = "N") -> None:
        super().__init__(rng, attack_budget=1)
        self.name = name
        self.variant_token = variant_token

    def generate(self, tokens, n_variants=1):
        outcomes = []
        for ordinal in range(n_variants):
            modified = list(tokens)
            modified[-1] = self.variant_token if ordinal % 2 == 0 else "O"
            outcomes.append(
                AttackOutcome(
                    original_tokens=tuple(tokens),
                    adversarial_tokens=tuple(modified),
                    attack_type=self.name,
                    edit_positions=(len(tokens) - 1,),
                )
            )
        return outcomes


def make_record(sample_id: str, drift: float, *, attack_type="randomization"):
    return AttackRecord(
        attack_id=f"{sample_id}:{attack_type}",
        sample_id=sample_id,
        original_representation="CC",
        adversarial_representation="CN",
        attack_type=attack_type,
        attack_budget=1,
        number_of_changes=1,
        edited_positions=(1,),
        validity_status="valid",
        plausibility_status="plausible",
        signed_prediction_drift=drift,
        absolute_prediction_drift=abs(drift),
    )


def test_attack_taxonomy_never_calls_chemical_edits_label_preserving() -> None:
    assert ATTACK_TAXONOMY["randomization"] is AttackCategory.REPRESENTATION_PRESERVING
    for family in ("substitution", "insertion", "deletion", "rearrangement"):
        assert ATTACK_TAXONOMY[family] is AttackCategory.TARGET_CHANGING
    assert ATTACK_TAXONOMY["probabilistic_mcmc"] is AttackCategory.STRESS_TEST


def test_success_rate_only_reported_for_representation_control() -> None:
    record = make_record("s1", 2.0)
    criterion = SuccessCriterion(min_abs_drift=1.0)
    control = summarize_attack_records(
        [record],
        category=AttackCategory.REPRESENTATION_PRESERVING,
        criterion=criterion,
        bootstrap_seed=1,
        bootstrap_resamples=10,
        bootstrap_min_samples=20,
    )
    stress = summarize_attack_records(
        [record],
        category=AttackCategory.TARGET_CHANGING,
        criterion=criterion,
        bootstrap_seed=1,
        bootstrap_resamples=10,
        bootstrap_min_samples=20,
    )
    assert control["attack_success_rate"] == 1.0
    assert control["stress_threshold_exceedance_rate"] is None
    assert stress["attack_success_rate"] is None
    assert stress["stress_threshold_exceedance_rate"] == 1.0


def test_cluster_bootstrap_is_omitted_for_tiny_sample_count() -> None:
    summary = summarize_attack_records(
        [make_record(f"s{i}", float(i + 1)) for i in range(5)],
        category=AttackCategory.REPRESENTATION_PRESERVING,
        criterion=SuccessCriterion(),
        bootstrap_seed=1,
        bootstrap_resamples=100,
        bootstrap_min_samples=20,
    )
    assert all(
        interval is None
        for interval in summary["bootstrap_confidence_intervals"].values()
    )
    assert summary["max_drift_ci"] is None


def test_paired_delta_bootstraps_source_samples_not_candidates() -> None:
    original = [make_record(f"s{i}", float(i + 2)) for i in range(25)]
    defended = [make_record(f"s{i}", float(i + 1)) for i in range(25)]
    pairs, summary = paired_defense_delta(
        original,
        defended,
        bootstrap_seed=4,
        bootstrap_resamples=100,
        bootstrap_min_samples=20,
    )
    assert len(pairs) == 25
    assert summary["mean_defense_delta"] == pytest.approx(-1.0)
    assert summary["mean_defense_delta_ci"]["n_source_samples"] == 25
    assert summary["significance_test"] is None


def test_paired_delta_excludes_invalid_candidates() -> None:
    valid_before = make_record("valid", 2.0)
    valid_after = make_record("valid", 1.0)
    invalid_before = AttackRecord.from_dict({
        **make_record("invalid", 100.0).to_dict(),
        "validity_status": "invalid_representation",
    })
    invalid_after = AttackRecord.from_dict({
        **make_record("invalid", 0.0).to_dict(),
        "validity_status": "invalid_representation",
    })
    pairs, summary = paired_defense_delta(
        [valid_before, invalid_before], [valid_after, invalid_after],
        bootstrap_seed=3, bootstrap_resamples=10, bootstrap_min_samples=20,
    )
    assert summary["n_pairs"] == 1
    assert summary["mean_defense_delta"] == -1.0
    assert pairs[0]["included_in_paired_delta"] is True
    assert pairs[1]["included_in_paired_delta"] is False
    assert pairs[1]["defense_delta"] is None


def test_closed_loop_persists_matrix_pairs_metadata_and_refuses_overwrite(tmp_path) -> None:
    run_dir = tmp_path / "phase4"
    clean_checkpoint = tmp_path / "clean.pt"
    clean_checkpoint.write_bytes(b"clean checkpoint")
    input_file = tmp_path / "input.csv"
    input_file.write_text("input\n")

    defenses = [
        DefenseSpec("clean", ()),
        DefenseSpec("randomization", ("randomization",)),
        DefenseSpec("mcmc", ("probabilistic_mcmc",)),
        DefenseSpec("mixed", ("randomization", "substitution")),
    ]
    families = [
        "randomization",
        "probabilistic_mcmc",
        "substitution",
        "insertion",
        "deletion",
        "rearrangement",
    ]

    def attack_factory(family, predictor, rng):
        token = "N"
        if family == "probabilistic_mcmc" and predictor.scale < 1.0:
            token = "O"
        return NamedAttack(rng, family, token)

    def build_augmented(defense, records, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("defense,n\n" + f"{defense.name},{len(records)}\n")
        return output_path

    scales = {"randomization": 0.9, "mcmc": 0.8, "mixed": 0.7}

    def train_defended(defense, augmented_path, model_dir, seed):
        assert augmented_path.exists() and seed >= 0
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").write_bytes(defense.name.encode())
        return ScalePredictor(scales[defense.name])

    experiment = ClosedLoopExperiment(
        run_dir,
        settings=ClosedLoopSettings(
            seed=12,
            n_train_variants=1,
            n_eval_variants=1,
            bootstrap_resamples=50,
            bootstrap_min_samples=20,
            check_plausibility=False,
        ),
        criterion=SuccessCriterion(min_abs_drift=0.1),
        attack_families=families,
        defenses=defenses,
        attack_factory=attack_factory,
    )
    evaluation = [(f"s{i}", "CC", 140.0) for i in range(25)]
    summary = experiment.run(
        clean_predictor=ScalePredictor(1.0),
        clean_checkpoint=clean_checkpoint,
        train_samples=[("train-1", "CC")],
        evaluation_examples=evaluation,
        build_augmented_dataset=build_augmented,
        train_defended=train_defended,
        config_snapshot={"lineage": "synthetic-test"},
        input_artifacts=[input_file],
    )

    assert summary["attacks"]["randomization"]["clean"][
        "success_rate_is_scientifically_meaningful"
    ] is True
    assert summary["attacks"]["substitution"]["clean"][
        "attack_success_rate"
    ] is None
    assert summary["attacks"]["randomization"]["randomization"][
        "defense_training_relation"
    ] == "same_attack_family"
    assert summary["attacks"]["insertion"]["randomization"][
        "defense_training_relation"
    ] == "different_attack_family"
    assert summary["attacks"]["substitution"]["mixed"][
        "defense_training_relation"
    ] == "mixed_defense_contains_attack"
    assert (run_dir / "paired_results/mixed/randomization.jsonl").exists()
    assert not (run_dir / "paired_results/mixed/probabilistic_mcmc.jsonl").exists()
    metadata = json.loads((run_dir / "reproducibility_metadata.json").read_text())
    assert set(metadata["checkpoint_hashes"]) == {
        "clean",
        "randomization",
        "mcmc",
        "mixed",
    }
    assert metadata["significance_tests"] is None
    assert metadata["training_bank_policy"].startswith("Each attack family")
    randomization_bank = (run_dir / "training_attack_records/randomization.jsonl").read_text()
    assert randomization_bank
    assert randomization_bank in (run_dir / "training_attacks/randomization.jsonl").read_text()
    assert randomization_bank in (run_dir / "training_attacks/mixed.jsonl").read_text()
    with (run_dir / "robustness_transfer_matrix.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["attack"] for row in rows] == [
        "randomization",
        "MCMC",
        "substitution",
        "insertion",
        "deletion",
        "rearrangement",
    ]
    assert list(rows[0]) == ["attack", "clean", "randomization", "MCMC", "mixed"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.run(
            clean_predictor=ScalePredictor(1.0),
            clean_checkpoint=clean_checkpoint,
            train_samples=[],
            evaluation_examples=[],
            build_augmented_dataset=build_augmented,
            train_defended=train_defended,
            config_snapshot={},
            input_artifacts=[],
        )


def test_shipped_phase4_config_declares_complete_matrix_and_label_policy() -> None:
    import yaml

    config = yaml.safe_load(Path("configs/closed_loop.yaml").read_text())
    assert config["lineage"] == "polyverse_bandgap"
    assert list(config["defenses"]) == ["clean", "randomization", "mcmc", "mixed"]
    assert config["attacks"]["families"] == [
        "randomization",
        "probabilistic_mcmc",
        "substitution",
        "insertion",
        "deletion",
        "rearrangement",
    ]
    assert config["scientific_policy"]["representation_preserving_label"] == "measured_target"
    assert "teacher_prediction" in config["scientific_policy"]["target_changing_label"]
    assert config["evaluation_protocol"]["prohibit_test_tuning"] is True


def test_graph_equivalence_gate_accepts_randomized_smiles_not_changed_graph() -> None:
    pytest.importorskip("rdkit")
    assert canonical_graph_equivalent("[*]CCO[*]", "O(CC[*])[*]")
    assert not canonical_graph_equivalent("[*]CCO[*]", "[*]CCN[*]")
