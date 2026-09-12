"""Phase 4 closed-loop adversarial evaluation.

This module owns scientific taxonomy, clustered bootstrap summaries, paired
robustness comparisons and immutable artifact layout. Model construction and
dataframe-specific training remain injected adapters so the existing training
implementation is reused rather than copied.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..attacks.base import BaseAttack
from ..attacks.generator import PredictorProtocol
from ..evaluation.attack_metrics import SuccessCriterion
from ..evaluation.records import AttackRecord
from .pipeline import AttackCandidate, generate_candidate_bank, score_candidate_bank


class AttackCategory(str, Enum):
    REPRESENTATION_PRESERVING = "representation_preserving_control"
    TARGET_CHANGING = "chemically_valid_target_changing_perturbation"
    STRESS_TEST = "adversarial_stress_test"


ATTACK_TAXONOMY: dict[str, AttackCategory] = {
    "randomization": AttackCategory.REPRESENTATION_PRESERVING,
    "substitution": AttackCategory.TARGET_CHANGING,
    "insertion": AttackCategory.TARGET_CHANGING,
    "deletion": AttackCategory.TARGET_CHANGING,
    "rearrangement": AttackCategory.TARGET_CHANGING,
    "probabilistic_mcmc": AttackCategory.STRESS_TEST,
}

ATTACK_DISPLAY_NAMES = {"probabilistic_mcmc": "MCMC"}
DEFENSE_DISPLAY_NAMES = {"mcmc": "MCMC"}


def stable_seed(root_seed: int, *parts: str) -> int:
    payload = ":".join([str(root_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: np.ndarray, q: float) -> float | None:
    return float(np.quantile(values, q)) if values.size else None


def _cluster_bootstrap_ci(
    values_by_sample: Mapping[str, Sequence[float]],
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    n_resamples: int,
    min_samples: int,
) -> dict[str, Any] | None:
    sample_ids = sorted(values_by_sample)
    if len(sample_ids) < min_samples or n_resamples < 1:
        return None
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        selected = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        values = np.concatenate(
            [np.asarray(values_by_sample[sample_id], dtype=float) for sample_id in selected]
        )
        estimates.append(float(statistic(values)))
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return {
        "method": "source-polymer clustered percentile bootstrap",
        "confidence_level": 0.95,
        "n_source_samples": len(sample_ids),
        "n_resamples": n_resamples,
        "low": float(low),
        "high": float(high),
    }


def summarize_attack_records(
    records: Sequence[AttackRecord],
    *,
    category: AttackCategory,
    criterion: SuccessCriterion,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    bootstrap_min_samples: int,
) -> dict[str, Any]:
    valid = [record for record in records if record.validity_status == "valid"]
    drifts = np.asarray(
        [
            record.absolute_prediction_drift
            for record in valid
            if record.absolute_prediction_drift is not None
        ],
        dtype=float,
    )
    valid_by_sample: dict[str, list[float]] = defaultdict(list)
    validity_by_sample: dict[str, list[float]] = defaultdict(list)
    success_by_sample: dict[str, list[float]] = defaultdict(list)
    for record in records:
        validity_by_sample[record.sample_id].append(
            float(record.validity_status == "valid")
        )
    for record in valid:
        if record.absolute_prediction_drift is not None:
            valid_by_sample[record.sample_id].append(record.absolute_prediction_drift)
        success_by_sample[record.sample_id].append(float(criterion.is_success(record)))

    metric_functions: dict[str, Callable[[np.ndarray], float]] = {
        "mean_abs_drift": lambda values: float(np.mean(values)),
        "median_abs_drift": lambda values: float(np.median(values)),
        "p90_abs_drift": lambda values: float(np.quantile(values, 0.90)),
        "p95_abs_drift": lambda values: float(np.quantile(values, 0.95)),
    }
    confidence_intervals = {
        name: _cluster_bootstrap_ci(
            valid_by_sample,
            function,
            seed=stable_seed(bootstrap_seed, name),
            n_resamples=bootstrap_resamples,
            min_samples=bootstrap_min_samples,
        )
        for name, function in metric_functions.items()
    }
    confidence_intervals["attack_validity_rate"] = _cluster_bootstrap_ci(
        validity_by_sample,
        lambda values: float(np.mean(values)),
        seed=stable_seed(bootstrap_seed, "attack_validity_rate"),
        n_resamples=bootstrap_resamples,
        min_samples=bootstrap_min_samples,
    )

    success_meaningful = category is AttackCategory.REPRESENTATION_PRESERVING
    successful = [record for record in valid if criterion.is_success(record)]
    confidence_intervals[
        "attack_success_rate"
        if success_meaningful
        else "stress_threshold_exceedance_rate"
    ] = _cluster_bootstrap_ci(
        success_by_sample,
        lambda values: float(np.mean(values)),
        seed=stable_seed(bootstrap_seed, "threshold_rate"),
        n_resamples=bootstrap_resamples,
        min_samples=bootstrap_min_samples,
    )
    return {
        "attack_category": category.value,
        "n_total": len(records),
        "n_valid": len(valid),
        "n_source_samples": len({record.sample_id for record in records}),
        "attack_validity_rate": len(valid) / len(records) if records else None,
        "mean_abs_drift": float(np.mean(drifts)) if drifts.size else None,
        "median_abs_drift": float(np.median(drifts)) if drifts.size else None,
        "p90_abs_drift": _quantile(drifts, 0.90),
        "p95_abs_drift": _quantile(drifts, 0.95),
        "max_abs_drift": float(np.max(drifts)) if drifts.size else None,
        "attack_success_rate": (
            len(successful) / len(valid) if success_meaningful and valid else None
        ),
        "stress_threshold_exceedance_rate": (
            len(successful) / len(valid) if not success_meaningful and valid else None
        ),
        "success_rate_is_scientifically_meaningful": success_meaningful,
        "criterion": asdict(criterion),
        "bootstrap_confidence_intervals": confidence_intervals,
        "bootstrap_note": (
            "Intervals resample source polymers, not correlated candidates. "
            "They are descriptive uncertainty intervals, not significance tests."
            if any(value is not None for value in confidence_intervals.values())
            else f"UNAVAILABLE: fewer than {bootstrap_min_samples} source samples or no resamples"
        ),
        "max_drift_ci": None,
        "max_drift_ci_note": "Not reported: a bootstrap CI for an observed maximum is not reliable here.",
    }


def evaluate_clean_mae(
    predictor: PredictorProtocol,
    examples: Sequence[tuple[str, str, float]],
) -> dict[str, Any]:
    predictions = predictor.predict([representation for _, representation, _ in examples])
    if len(predictions) != len(examples):
        raise ValueError("predictor returned a different number of predictions than inputs")
    targets = np.asarray([target for _, _, target in examples], dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    errors = np.abs(predictions - targets)
    return {
        "n": len(examples),
        "mae": float(np.mean(errors)) if errors.size else None,
        "per_example": [
            {
                "sample_id": sample_id,
                "representation": representation,
                "target": float(target),
                "prediction": float(prediction),
                "absolute_error": float(error),
            }
            for (sample_id, representation, target), prediction, error in zip(
                examples, predictions, errors
            )
        ],
    }


def paired_defense_delta(
    original: Sequence[AttackRecord],
    defended: Sequence[AttackRecord],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    bootstrap_min_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_by_id = {record.attack_id: record for record in original}
    defended_by_id = {record.attack_id: record for record in defended}
    if list(original_by_id) != list(defended_by_id):
        raise ValueError("paired defense delta requires identical ordered attack IDs")
    pairs: list[dict[str, Any]] = []
    delta_by_sample: dict[str, list[float]] = defaultdict(list)
    for attack_id in original_by_id:
        before, after = original_by_id[attack_id], defended_by_id[attack_id]
        if (
            before.original_representation != after.original_representation
            or before.adversarial_representation != after.adversarial_representation
        ):
            raise ValueError(f"paired candidate payload differs for {attack_id}")
        before_drift = before.absolute_prediction_drift
        after_drift = after.absolute_prediction_drift
        eligible = (
            before.validity_status == "valid"
            and after.validity_status == "valid"
            and before_drift is not None
            and after_drift is not None
        )
        delta = (
            float(after_drift - before_drift)
            if eligible
            else None
        )
        pairs.append(
            {
                "attack_id": attack_id,
                "sample_id": before.sample_id,
                "attack_type": before.attack_type,
                "original_representation": before.original_representation,
                "adversarial_representation": before.adversarial_representation,
                "original_model_abs_drift": before_drift,
                "defended_model_abs_drift": after_drift,
                "defense_delta": delta,
                "included_in_paired_delta": eligible,
                "validity_status": before.validity_status,
            }
        )
        if delta is not None:
            delta_by_sample[before.sample_id].append(delta)
    deltas = np.asarray(
        [pair["defense_delta"] for pair in pairs if pair["defense_delta"] is not None],
        dtype=float,
    )
    ci = _cluster_bootstrap_ci(
        delta_by_sample,
        lambda values: float(np.mean(values)),
        seed=bootstrap_seed,
        n_resamples=bootstrap_resamples,
        min_samples=bootstrap_min_samples,
    )
    return pairs, {
        "paired": True,
        "n_pairs": len(deltas),
        "pairing_population": "candidates valid under both models",
        "mean_defense_delta": float(np.mean(deltas)) if deltas.size else None,
        "median_defense_delta": float(np.median(deltas)) if deltas.size else None,
        "mean_defense_delta_ci": ci,
        "interpretation": "negative values indicate lower defended-model drift",
        "significance_test": None,
    }


@dataclass(frozen=True, slots=True)
class DefenseSpec:
    name: str
    training_attacks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClosedLoopSettings:
    seed: int
    n_train_variants: int
    n_eval_variants: int
    bootstrap_resamples: int = 2000
    bootstrap_min_samples: int = 20
    check_plausibility: bool = True


class ClosedLoopExperiment:
    """Execute one immutable cross-attack robustness experiment."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        settings: ClosedLoopSettings,
        criterion: SuccessCriterion,
        attack_families: Sequence[str],
        defenses: Sequence[DefenseSpec],
        attack_factory: Callable[
            [str, PredictorProtocol, np.random.Generator], BaseAttack
        ],
    ) -> None:
        unknown = set(attack_families) - ATTACK_TAXONOMY.keys()
        if unknown:
            raise ValueError(f"unclassified attack families: {sorted(unknown)}")
        if not attack_families:
            raise ValueError("at least one attack family is required")
        defense_names = [defense.name for defense in defenses]
        if defense_names != ["clean", "randomization", "mcmc", "mixed"]:
            raise ValueError(
                "defenses must be ordered as clean, randomization, mcmc, mixed"
            )
        unknown_training = {
            family
            for defense in defenses
            for family in defense.training_attacks
            if family not in ATTACK_TAXONOMY
        }
        if unknown_training:
            raise ValueError(
                f"unclassified defense training attacks: {sorted(unknown_training)}"
            )
        self.run_dir = Path(run_dir)
        self.settings = settings
        self.criterion = criterion
        self.attack_families = tuple(attack_families)
        self.defenses = tuple(defenses)
        self.attack_factory = attack_factory

    def _write_json(self, relative: str, payload: Any) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def _write_jsonl(self, relative: str, rows: Sequence[dict[str, Any]]) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return path

    def run(
        self,
        *,
        clean_predictor: PredictorProtocol,
        clean_checkpoint: Path,
        train_samples: Sequence[tuple[str, str]],
        evaluation_examples: Sequence[tuple[str, str, float]],
        build_augmented_dataset: Callable[
            [DefenseSpec, Sequence[AttackRecord], Path], Path
        ],
        train_defended: Callable[[DefenseSpec, Path, Path, int], PredictorProtocol],
        config_snapshot: Mapping[str, Any],
        input_artifacts: Sequence[Path],
    ) -> dict[str, Any]:
        if self.run_dir.exists():
            raise FileExistsError(f"refusing to overwrite run directory: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        self._write_json("config_snapshot.json", dict(config_snapshot))

        evaluation_samples = [
            (sample_id, representation)
            for sample_id, representation, _ in evaluation_examples
        ]
        predictors: dict[str, PredictorProtocol] = {"clean": clean_predictor}
        checkpoint_paths: dict[str, Path] = {"clean": clean_checkpoint}

        # Generate each training family once against the clean model. Specialized
        # and mixed defenses then consume the identical family bank, avoiding a
        # candidate-distribution confound and redundant model scoring.
        training_families = tuple(dict.fromkeys(
            family for defense in self.defenses for family in defense.training_attacks
        ))
        training_records_by_family: dict[str, list[AttackRecord]] = {}
        for family in training_families:
            attack_seed = stable_seed(self.settings.seed, "train", family)
            attack = self.attack_factory(
                family, clean_predictor, np.random.default_rng(attack_seed)
            )
            candidates = generate_candidate_bank(
                train_samples, [attack], n_variants=self.settings.n_train_variants,
                seed=attack_seed,
            )
            self._write_jsonl(
                f"training_attack_candidates/{family}.jsonl",
                [candidate.to_dict() for candidate in candidates],
            )
            records = score_candidate_bank(
                candidates, clean_predictor, criterion=self.criterion,
                check_plausibility=self.settings.check_plausibility,
            )
            training_records_by_family[family] = records
            self._write_jsonl(
                f"training_attack_records/{family}.jsonl",
                [record.to_dict() for record in records],
            )

        # The augmentation callback assigns measured targets only to verified
        # representation-preserving controls and teacher predictions otherwise.
        for defense in self.defenses:
            if defense.name == "clean":
                continue
            training_records = [
                record for family in defense.training_attacks
                for record in training_records_by_family[family]
            ]
            self._write_jsonl(
                f"training_attacks/{defense.name}.jsonl",
                [record.to_dict() for record in training_records],
            )
            augmented_path = build_augmented_dataset(
                defense,
                training_records,
                self.run_dir / "training_data" / f"{defense.name}.csv",
            )
            defense_seed = stable_seed(self.settings.seed, "defense", defense.name)
            model_dir = self.run_dir / "models" / defense.name
            predictors[defense.name] = train_defended(
                defense, augmented_path, model_dir, defense_seed
            )
            checkpoint_paths[defense.name] = model_dir / "model.pt"

        summaries: dict[str, Any] = {
            "schema_version": "phase4.closed_loop.v1",
            "scientific_scope": {
                "representation_preserving": ["randomization"],
                "target_changing": [
                    "substitution",
                    "insertion",
                    "deletion",
                    "rearrangement",
                ],
                "stress_tests": ["probabilistic_mcmc"],
                "warning": (
                    "Target-changing and stress-test attacks do not inherit measured "
                    "Tg/bandgap labels. Their threshold exceedance is not target error."
                ),
            },
            "clean_performance": {},
            "attacks": {},
        }
        clean_mae: dict[str, float | None] = {}
        for defense_name, predictor in predictors.items():
            clean_eval = evaluate_clean_mae(predictor, evaluation_examples)
            clean_mae[defense_name] = clean_eval["mae"]
            summaries["clean_performance"][defense_name] = {
                "n": clean_eval["n"],
                "mae": clean_eval["mae"],
            }
            self._write_jsonl(
                f"clean_predictions/{defense_name}.jsonl",
                clean_eval["per_example"],
            )
        for defense_name in predictors:
            summaries["clean_performance"][defense_name]["tradeoff_vs_clean"] = (
                clean_mae[defense_name] - clean_mae["clean"]
                if clean_mae[defense_name] is not None
                and clean_mae["clean"] is not None
                else None
            )

        fixed_banks: dict[str, list[AttackCandidate]] = {}
        records_by_cell: dict[tuple[str, str], list[AttackRecord]] = {}
        for family in self.attack_families:
            adaptive = family == "probabilistic_mcmc"
            if not adaptive:
                attack_seed = stable_seed(self.settings.seed, "evaluate", family)
                attack = self.attack_factory(
                    family, clean_predictor, np.random.default_rng(attack_seed)
                )
                fixed_banks[family] = generate_candidate_bank(
                    evaluation_samples,
                    [attack],
                    n_variants=self.settings.n_eval_variants,
                    seed=attack_seed,
                )
                self._write_jsonl(
                    f"attack_candidates/{family}.jsonl",
                    [candidate.to_dict() for candidate in fixed_banks[family]],
                )

            for defense_name, predictor in predictors.items():
                if adaptive:
                    attack_seed = stable_seed(
                        self.settings.seed, "evaluate", family, defense_name
                    )
                    attack = self.attack_factory(
                        family, predictor, np.random.default_rng(attack_seed)
                    )
                    candidates = generate_candidate_bank(
                        evaluation_samples,
                        [attack],
                        n_variants=self.settings.n_eval_variants,
                        seed=attack_seed,
                    )
                    self._write_jsonl(
                        f"attack_candidates/{family}/{defense_name}.jsonl",
                        [candidate.to_dict() for candidate in candidates],
                    )
                else:
                    candidates = fixed_banks[family]
                records = score_candidate_bank(
                    candidates,
                    predictor,
                    criterion=self.criterion,
                    check_plausibility=self.settings.check_plausibility,
                )
                records_by_cell[(family, defense_name)] = records
                self._write_jsonl(
                    f"raw_attack_records/{defense_name}/{family}.jsonl",
                    [record.to_dict() for record in records],
                )
                summaries["attacks"].setdefault(family, {})[defense_name] = {
                    **summarize_attack_records(
                        records,
                        category=ATTACK_TAXONOMY[family],
                        criterion=self.criterion,
                        bootstrap_seed=stable_seed(
                            self.settings.seed, "bootstrap", family, defense_name
                        ),
                        bootstrap_resamples=self.settings.bootstrap_resamples,
                        bootstrap_min_samples=self.settings.bootstrap_min_samples,
                    ),
                    "candidate_mode": "adaptive_unpaired" if adaptive else "fixed_paired",
                }

        matrix_rows: list[dict[str, Any]] = []
        defense_by_name = {defense.name: defense for defense in self.defenses}
        for family in self.attack_families:
            original_summary = summaries["attacks"][family]["clean"]
            original_drift = original_summary["mean_abs_drift"]
            for defense_name in predictors:
                cell = summaries["attacks"][family][defense_name]
                defended_drift = cell["mean_abs_drift"]
                paired = family != "probabilistic_mcmc"
                if defense_name == "clean":
                    delta_summary = {
                        "paired": paired,
                        "mean_defense_delta": 0.0,
                        "mean_defense_delta_ci": None,
                    }
                elif paired:
                    pairs, delta_summary = paired_defense_delta(
                        records_by_cell[(family, "clean")],
                        records_by_cell[(family, defense_name)],
                        bootstrap_seed=stable_seed(
                            self.settings.seed, "delta", family, defense_name
                        ),
                        bootstrap_resamples=self.settings.bootstrap_resamples,
                        bootstrap_min_samples=self.settings.bootstrap_min_samples,
                    )
                    self._write_jsonl(
                        f"paired_results/{defense_name}/{family}.jsonl", pairs
                    )
                else:
                    delta_summary = {
                        "paired": False,
                        "mean_defense_delta": (
                            defended_drift - original_drift
                            if defended_drift is not None and original_drift is not None
                            else None
                        ),
                        "mean_defense_delta_ci": None,
                        "note": "Adaptive MCMC candidates differ by target model; no paired inference.",
                    }
                robustness_transfer = (
                    1.0 - defended_drift / original_drift
                    if defended_drift is not None and original_drift not in (None, 0.0)
                    else None
                )
                clean_tradeoff = (
                    clean_mae[defense_name] - clean_mae["clean"]
                    if clean_mae[defense_name] is not None
                    and clean_mae["clean"] is not None
                    else None
                )
                cell["defense_delta"] = delta_summary
                cell["robustness_transfer"] = robustness_transfer
                cell["clean_performance_tradeoff"] = clean_tradeoff
                if defense_name == "clean":
                    relation = "original_model_attacked"
                elif family in defense_by_name[defense_name].training_attacks:
                    relation = (
                        "mixed_defense_contains_attack"
                        if defense_name == "mixed"
                        else "same_attack_family"
                    )
                else:
                    relation = "different_attack_family"
                cell["defense_training_relation"] = relation
                matrix_rows.append(
                    {
                        "attack": family,
                        "attack_category": ATTACK_TAXONOMY[family].value,
                        "trained_defense": defense_name,
                        "mean_abs_drift": defended_drift,
                        "defense_delta": delta_summary["mean_defense_delta"],
                        "robustness_transfer": robustness_transfer,
                        "clean_mae": clean_mae[defense_name],
                        "clean_performance_tradeoff": clean_tradeoff,
                        "attack_validity_rate": cell["attack_validity_rate"],
                        "attack_success_rate": cell["attack_success_rate"],
                        "stress_threshold_exceedance_rate": cell[
                            "stress_threshold_exceedance_rate"
                        ],
                        "paired": delta_summary["paired"],
                        "defense_training_relation": relation,
                    }
                )

        cells_path = self.run_dir / "robustness_transfer_cells.csv"
        with cells_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]))
            writer.writeheader()
            writer.writerows(matrix_rows)
        matrix_path = self.run_dir / "robustness_transfer_matrix.csv"
        defense_names = list(predictors)
        with matrix_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "attack",
                    *[DEFENSE_DISPLAY_NAMES.get(name, name) for name in defense_names],
                ],
            )
            writer.writeheader()
            for family in self.attack_families:
                writer.writerow(
                    {
                        "attack": ATTACK_DISPLAY_NAMES.get(family, family),
                        **{
                            DEFENSE_DISPLAY_NAMES.get(defense_name, defense_name):
                            summaries["attacks"][family][defense_name]["mean_abs_drift"]
                            for defense_name in defense_names
                        },
                    }
                )
        self._write_json("summary.json", summaries)

        package_versions = {}
        for distribution in ("numpy", "pandas", "PyYAML", "rdkit", "torch"):
            try:
                package_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                package_versions[distribution] = None
        attack_seeds = {
            "training": {
                family: stable_seed(self.settings.seed, "train", family)
                for family in training_families
            },
            "evaluation": {
                family: (
                    {
                        defense_name: stable_seed(
                            self.settings.seed,
                            "evaluate",
                            family,
                            defense_name,
                        )
                        for defense_name in predictors
                    }
                    if family == "probabilistic_mcmc"
                    else stable_seed(self.settings.seed, "evaluate", family)
                )
                for family in self.attack_families
            },
        }
        metadata = {
            "schema_version": "phase4.reproducibility.v1",
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": package_versions,
            "source": config_snapshot.get("source", {}),
            "settings": asdict(self.settings),
            "derived_seeds": {
                defense.name: stable_seed(self.settings.seed, "defense", defense.name)
                for defense in self.defenses
                if defense.name != "clean"
            },
            "seed_derivation": "sha256(root_seed:stage[:defense]:attack), reduced modulo 2**32",
            "attack_seeds": attack_seeds,
            "input_hashes": {
                str(path): sha256_file(path) for path in input_artifacts
            },
            "checkpoint_hashes": {
                defense_name: sha256_file(path)
                for defense_name, path in checkpoint_paths.items()
            },
            "pairing_policy": (
                "Fixed attack candidates are shared across models. Model-guided MCMC "
                "is regenerated per model and reported as adaptive/unpaired."
            ),
            "training_bank_policy": (
                "Each attack family is generated once against the clean model and "
                "reused unchanged in every defense that includes that family."
            ),
            "matrix_value": "mean absolute prediction drift in target units",
            "significance_tests": None,
        }
        self._write_json("reproducibility_metadata.json", metadata)
        return summaries
