"""One composable path through attack and defense experiments.

The orchestration is deliberately model- and dataframe-agnostic. Existing
training and augmentation functions can be passed as callbacks, while attack
generation, validation, record serialization and paired comparison have one
implementation. A run directory is created exclusively and is never reused, so
historical experiment outputs cannot be silently overwritten.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..attacks.base import BaseAttack
from ..attacks.generator import PredictorProtocol
from ..data.tokenizer import tokenize
from ..evaluation.attack_metrics import SuccessCriterion, group_by_attack_type
from ..evaluation.records import AttackRecord, compute_drift, make_attack_id
from ..validation.pipeline import validate
from ..utils.io import to_jsonable


@dataclass(frozen=True, slots=True)
class AttackCandidate:
    """A model-independent attack candidate that can be scored repeatedly."""

    candidate_id: str
    sample_id: str
    original_representation: str
    adversarial_representation: str
    attack_type: str
    attack_budget: int
    number_of_changes: int
    edited_positions: tuple[int, ...]
    attack_params: dict[str, Any]
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["edited_positions"] = list(self.edited_positions)
        return to_jsonable(data)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_dir: Path
    candidates_path: Path
    clean_records_path: Path
    augmented_dataset_path: Path | None = None
    defended_records_path: Path | None = None
    comparison_path: Path | None = None


def _write_jsonl_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_candidate_bank(
    samples: Sequence[tuple[str, str]],
    attacks: Sequence[BaseAttack],
    *,
    n_variants: int,
    seed: int | None,
) -> list[AttackCandidate]:
    """Generate candidates once so every target model sees identical inputs."""
    candidates: list[AttackCandidate] = []
    for sample_id, representation in samples:
        tokens = tokenize(representation)
        for attack in attacks:
            metadata = attack.metadata()
            for ordinal, outcome in enumerate(
                attack.generate(tokens, n_variants=n_variants)
            ):
                candidate_id = make_attack_id(
                    sample_id, outcome.attack_type, seed, ordinal
                )
                candidates.append(
                    AttackCandidate(
                        candidate_id=candidate_id,
                        sample_id=sample_id,
                        original_representation=outcome.original_representation,
                        adversarial_representation=outcome.adversarial_representation,
                        attack_type=outcome.attack_type,
                        attack_budget=int(metadata.get("attack_budget", 1)),
                        number_of_changes=outcome.number_of_changes,
                        edited_positions=outcome.edit_positions,
                        attack_params={**metadata, **outcome.params},
                        seed=seed,
                    )
                )
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(
            "candidate IDs are not unique; do not configure duplicate attack names"
        )
    return candidates


def score_candidate_bank(
    candidates: Sequence[AttackCandidate],
    predictor: PredictorProtocol,
    *,
    criterion: SuccessCriterion,
    check_plausibility: bool = True,
) -> list[AttackRecord]:
    """Validate a frozen candidate bank and score it against one predictor."""
    if not candidates:
        return []

    representations = sorted(
        {
            representation
            for candidate in candidates
            for representation in (
                candidate.original_representation,
                candidate.adversarial_representation,
            )
        }
    )
    predictions = predictor.predict(representations)
    if len(predictions) != len(representations):
        raise ValueError(
            "predictor returned a different number of predictions than inputs"
        )
    prediction_by_representation = {
        representation: float(prediction)
        for representation, prediction in zip(representations, predictions)
    }

    records: list[AttackRecord] = []
    for candidate in candidates:
        validation = validate(
            candidate.adversarial_representation,
            check_plausibility=check_plausibility,
        )
        original_prediction = prediction_by_representation[
            candidate.original_representation
        ]
        adversarial_prediction = prediction_by_representation[
            candidate.adversarial_representation
        ]
        drift = compute_drift(original_prediction, adversarial_prediction)
        status = validation.status.value
        record = AttackRecord(
            attack_id=candidate.candidate_id,
            sample_id=candidate.sample_id,
            original_representation=candidate.original_representation,
            adversarial_representation=candidate.adversarial_representation,
            attack_type=candidate.attack_type,
            attack_budget=candidate.attack_budget,
            number_of_changes=candidate.number_of_changes,
            edited_positions=candidate.edited_positions,
            validity_status=status,
            plausibility_status=(
                "plausible"
                if status == "valid"
                else "implausible" if status == "implausible" else "unchecked"
            ),
            original_prediction=original_prediction,
            adversarial_prediction=adversarial_prediction,
            signed_prediction_drift=drift,
            absolute_prediction_drift=abs(drift) if drift is not None else None,
            rejection_reasons=validation.rejection_reasons,
            plausibility_flags=validation.plausibility_flags,
            checks_skipped=validation.checks_skipped,
            attack_score=candidate.attack_params.get("attack_score"),
            chemical_constraint_score=candidate.attack_params.get(
                "chemical_constraint_score"
            ),
            sampling_probability=candidate.attack_params.get(
                "sampling_probability"
            ),
            seed=candidate.seed,
            attack_params=candidate.attack_params,
        )
        records.append(
            AttackRecord.from_dict(
                {**record.to_dict(), "attack_success": criterion.is_success(record)}
            )
        )
    return records


def compare_paired_records(
    clean_records: Sequence[AttackRecord],
    defended_records: Sequence[AttackRecord],
    *,
    criterion: SuccessCriterion,
) -> dict[str, Any]:
    """Compare two score sets only when candidate identity and order match."""
    clean_ids = [record.attack_id for record in clean_records]
    defended_ids = [record.attack_id for record in defended_records]
    if clean_ids != defended_ids:
        raise ValueError("paired comparison requires identical ordered candidate IDs")
    for clean, defended in zip(clean_records, defended_records):
        clean_candidate = (
            clean.original_representation,
            clean.adversarial_representation,
            clean.attack_type,
        )
        defended_candidate = (
            defended.original_representation,
            defended.adversarial_representation,
            defended.attack_type,
        )
        if clean_candidate != defended_candidate:
            raise ValueError(
                f"candidate payload differs for paired ID {clean.attack_id!r}"
            )

    clean_summary = group_by_attack_type(clean_records, criterion)
    defended_summary = group_by_attack_type(defended_records, criterion)
    deltas: dict[str, dict[str, float]] = {}
    for attack_type in sorted(clean_summary.keys() & defended_summary.keys()):
        before = clean_summary[attack_type]
        after = defended_summary[attack_type]
        deltas[attack_type] = {
            "success_rate": float(after["success_rate"])
            - float(before["success_rate"]),
            "mean_abs_drift": float(after.get("mean_abs_drift", 0.0))
            - float(before.get("mean_abs_drift", 0.0)),
        }
    return {
        "paired": True,
        "n_candidates": len(clean_records),
        "criterion": asdict(criterion),
        "clean": clean_summary,
        "defended": defended_summary,
        "defended_minus_clean": deltas,
    }


class ExperimentPipeline:
    """Run an immutable clean -> attack -> defend -> paired re-attack workflow."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def run(
        self,
        *,
        samples: Sequence[tuple[str, str]],
        train_clean: Callable[[Path], PredictorProtocol],
        build_attacks: Callable[[PredictorProtocol], Sequence[BaseAttack]],
        criterion: SuccessCriterion,
        n_variants: int,
        seed: int | None,
        augment: Callable[[Sequence[AttackRecord], Path], Path] | None = None,
        train_defended: Callable[[Path, Path], PredictorProtocol] | None = None,
        check_plausibility: bool = True,
    ) -> ExperimentResult:
        if self.run_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite experiment directory: {self.run_dir}"
            )
        if (augment is None) != (train_defended is None):
            raise ValueError("augment and train_defended must be supplied together")
        self.run_dir.mkdir(parents=True)

        clean_predictor = train_clean(self.run_dir / "clean_model")
        candidates = generate_candidate_bank(
            samples,
            build_attacks(clean_predictor),
            n_variants=n_variants,
            seed=seed,
        )
        candidates_path = self.run_dir / "attack_candidates.jsonl"
        _write_jsonl_exclusive(
            candidates_path, [candidate.to_dict() for candidate in candidates]
        )

        clean_records = score_candidate_bank(
            candidates,
            clean_predictor,
            criterion=criterion,
            check_plausibility=check_plausibility,
        )
        clean_records_path = self.run_dir / "clean_attack_records.jsonl"
        _write_jsonl_exclusive(
            clean_records_path, [record.to_dict() for record in clean_records]
        )

        if augment is None or train_defended is None:
            return ExperimentResult(
                self.run_dir, candidates_path, clean_records_path
            )

        augmented_path = augment(
            clean_records, self.run_dir / "adversarial_train.csv"
        )
        defended_predictor = train_defended(
            augmented_path, self.run_dir / "defended_model"
        )
        defended_records = score_candidate_bank(
            candidates,
            defended_predictor,
            criterion=criterion,
            check_plausibility=check_plausibility,
        )
        defended_records_path = self.run_dir / "defended_attack_records.jsonl"
        _write_jsonl_exclusive(
            defended_records_path,
            [record.to_dict() for record in defended_records],
        )
        comparison = compare_paired_records(
            clean_records, defended_records, criterion=criterion
        )
        comparison_path = self.run_dir / "paired_robustness.json"
        _write_json_exclusive(comparison_path, comparison)
        return ExperimentResult(
            self.run_dir,
            candidates_path,
            clean_records_path,
            augmented_path,
            defended_records_path,
            comparison_path,
        )
