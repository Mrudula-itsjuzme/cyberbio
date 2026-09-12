#!/usr/bin/env python3
"""Run Phase 4 closed-loop adversarial evaluation.

The reusable experiment and statistics live under ``materials_adv.experiments``.
This script only adapts the checked-in dataframe, Transformer and attack classes.
It always requires a new output directory and never edits historical artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.probabilistic import ProbabilisticMCMCAttack
from materials_adv.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.evaluation.attack_metrics import SuccessCriterion
from materials_adv.experiments.closed_loop import (
    ATTACK_TAXONOMY,
    AttackCategory,
    ClosedLoopExperiment,
    ClosedLoopSettings,
    DefenseSpec,
)
from materials_adv.utils.config import load_config
from materials_adv.validation.representation import canonical_graph_equivalent


def preflight_phase4(
    *, run_dir: Path, dataframe: pd.DataFrame, splits: dict, vocab: list[str],
    phase4_cfg: dict, dataset_cfg: dict, model_cfg: dict, tokenizer_cfg: dict,
    clean_model_dir: Path,
) -> dict[str, object]:
    """Fail closed on lineage, split, preprocessing, and checkpoint mismatches."""
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {run_dir}")
    required = [
        clean_model_dir / "model.pt", clean_model_dir / "scaler.json",
        clean_model_dir / "metrics.json", Path(dataset_cfg["processed_dir"]) / "processed.csv",
        Path(dataset_cfg["processed_dir"]) / "splits.json",
        Path(dataset_cfg["processed_dir"]) / "vocab.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase 4 inputs: {missing}")
    memberships = {name: tuple(splits[name]) for name in ("train", "val", "test")}
    member_sets = {name: set(values) for name, values in memberships.items()}
    if any(member_sets[a] & member_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("train/val/test indices overlap")
    if set().union(*member_sets.values()) != set(range(len(dataframe))):
        raise ValueError("split indices do not cover the processed dataset exactly")
    for name, indices in memberships.items():
        if set(dataframe.iloc[list(indices)]["split"].astype(str)) != {name}:
            raise ValueError(f"processed split column disagrees with splits.json for {name}")
    if not splits.get("test_sealed", False) or not dataset_cfg["split"].get("test_sealed", False):
        raise ValueError("test split must be declared sealed")
    if phase4_cfg["evaluation_split"] == "test" and phase4_cfg["evaluation_protocol"]["test_use"] != "one_time_final_evaluation_after_validation_phase_freeze":
        raise ValueError("test evaluation requires an explicit final-evaluation protocol")
    if phase4_cfg["lineage"] != dataset_cfg["name"] or dataset_cfg["target_units"] != "eV":
        raise ValueError("Phase 4 must use the active polyVERSE eV lineage")
    if set(dataframe["units"].astype(str)) != {dataset_cfg["target_units"]}:
        raise ValueError("processed target units disagree with dataset config")
    if len(vocab) != tokenizer_cfg["vocab_size"] or len(vocab) != model_cfg["architecture"]["vocab_size"]:
        raise ValueError("vocabulary size disagrees across artifact/tokenizer/model config")

    from materials_adv.data.scaler import TargetScaler
    from materials_adv.data.tokenizer import tokenize
    import torch
    metrics = json.loads((clean_model_dir / "metrics.json").read_text())
    scaler = TargetScaler.load(clean_model_dir / "scaler.json")
    if metrics["n_train"] != len(memberships["train"]) or metrics["n_val"] != len(memberships["val"]) or metrics["n_test"] != len(memberships["test"]):
        raise ValueError("checkpoint metrics split counts are incompatible")
    if metrics["scaler"] != {"mean": scaler.mean, "std": scaler.std}:
        raise ValueError("checkpoint metrics and scaler artifact disagree")
    state = torch.load(clean_model_dir / "model.pt", map_location="cpu", weights_only=True)
    architecture = model_cfg["architecture"]
    if tuple(state["embedding.weight"].shape) != (len(vocab) + 1, architecture["d_model"]):
        raise ValueError("checkpoint embedding shape is incompatible with vocabulary/config")
    if tuple(state["pos_encoder.weight"].shape) != (architecture["max_seq_len"], architecture["d_model"]):
        raise ValueError("checkpoint positional embedding shape is incompatible")
    max_tokens = max(len(tokenize(value)) for value in dataframe.original_representation.astype(str))
    if max_tokens > architecture["max_seq_len"]:
        raise ValueError("dataset contains sequences longer than model maximum")

    n_train, n_eval = len(memberships["train"]), len(memberships[phase4_cfg["evaluation_split"]])
    fixed_families = sum(f != "probabilistic_mcmc" for f in phase4_cfg["attacks"]["families"])
    max_training_rows = {
        name: n_train * (1 + len(families) * phase4_cfg["attacks"]["training_variants"])
        for name, families in phase4_cfg["defenses"].items() if name != "clean"
    }
    return {
        "status": "passed", "lineage": phase4_cfg["lineage"],
        "target_units": dataset_cfg["target_units"], "evaluation_split": phase4_cfg["evaluation_split"],
        "split_counts": {name: len(values) for name, values in memberships.items()},
        "split_overlap_count": 0, "training_uses_test_examples": False,
        "test_use": phase4_cfg["evaluation_protocol"]["test_use"],
        "vocab_size": len(vocab), "max_token_length": max_tokens,
        "checkpoint_embedding_shape": list(state["embedding.weight"].shape),
        "scaler": {"mean": scaler.mean, "std": scaler.std},
        "compute_upper_bounds": {
            "unique_training_families": len({f for values in phase4_cfg["defenses"].values() for f in values}),
            "max_augmented_rows_by_defense": max_training_rows,
            "fixed_evaluation_candidates": n_eval * fixed_families * phase4_cfg["attacks"]["evaluation_variants"],
            "adaptive_mcmc_proposal_steps": n_eval * len(phase4_cfg["defenses"]) * phase4_cfg["attacks"]["mcmc_steps"],
            "defended_models_to_train": 3,
            "max_epochs_per_model": model_cfg["training"]["epochs"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="New, non-existing output directory")
    parser.add_argument("--config", default="configs/closed_loop.yaml")
    parser.add_argument(
        "--clean-model-dir", default="results/models/transformer_regressor"
    )
    return parser.parse_args()


def load_predictor(
    model_dir: Path,
    *,
    model_cfg: dict,
    vocab: list[str],
    target_units: str,
) -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Phase 4 execution requires the optional model dependency: install "
            "the project with the 'model' extra or provide the configured torch runtime"
        ) from exc
    from materials_adv.data.scaler import TargetScaler
    from materials_adv.models.regression import TransformerRegressor
    from materials_adv.models.transformer import TransformerRegressorModel

    device = torch.device(model_cfg["device"])
    architecture = model_cfg["architecture"]
    model = TransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=architecture["d_model"],
        n_layers=architecture["n_layers"],
        n_heads=architecture["n_heads"],
        dim_feedforward=architecture["dim_feedforward"],
        dropout=architecture["dropout"],
        max_seq_len=architecture["max_seq_len"],
        pooling=architecture["pooling"],
    ).to(device)
    model.load_state_dict(
        torch.load(model_dir / "model.pt", weights_only=True, map_location=device)
    )
    scaler = TargetScaler.load(model_dir / "scaler.json")
    return TransformerRegressor(model, vocab, scaler, target_units)


def git_metadata() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return {"git_commit": commit, "working_tree_dirty": dirty}


def main() -> None:
    args = parse_args()
    phase4_cfg = yaml.safe_load(Path(args.config).read_text())
    dataset_cfg = load_config("dataset")
    model_cfg = load_config("model")
    attack_cfg = load_config("attack")
    tokenizer_cfg = load_config("tokenizer")
    if phase4_cfg["lineage"] != dataset_cfg["name"]:
        raise ValueError(
            "closed-loop lineage does not match configs/dataset.yaml: "
            f"{phase4_cfg['lineage']!r} != {dataset_cfg['name']!r}"
        )

    processed_dir = Path(dataset_cfg["processed_dir"])
    processed_path = processed_dir / "processed.csv"
    splits_path = processed_dir / "splits.json"
    vocab_path = processed_dir / "vocab.json"
    dataframe = pd.read_csv(processed_path)
    splits = json.loads(splits_path.read_text())
    vocab = json.loads(vocab_path.read_text())
    representation_column = "original_representation"
    target_column = "property_value"

    train_indices = splits["train"]
    evaluation_indices = splits[phase4_cfg["evaluation_split"]]
    train_samples = [
        (f"sample_{index}", str(dataframe.iloc[index][representation_column]))
        for index in train_indices
    ]
    evaluation_examples = [
        (
            f"sample_{index}",
            str(dataframe.iloc[index][representation_column]),
            float(dataframe.iloc[index][target_column]),
        )
        for index in evaluation_indices
    ]
    target_by_sample = {
        f"sample_{index}": float(dataframe.iloc[index][target_column])
        for index in train_indices
    }
    clean_train = dataframe.iloc[train_indices].copy()

    clean_model_dir = Path(args.clean_model_dir)
    preflight = preflight_phase4(
        run_dir=Path(args.run_dir), dataframe=dataframe, splits=splits, vocab=vocab,
        phase4_cfg=phase4_cfg, dataset_cfg=dataset_cfg, model_cfg=model_cfg,
        tokenizer_cfg=tokenizer_cfg, clean_model_dir=clean_model_dir,
    )
    print(json.dumps({"phase4_preflight": preflight}, indent=2))
    clean_predictor = load_predictor(
        clean_model_dir,
        model_cfg=model_cfg,
        vocab=vocab,
        target_units=dataset_cfg["target_units"],
    )

    mcmc_steps = int(phase4_cfg["attacks"]["mcmc_steps"])
    mcmc_temperature = float(phase4_cfg["attacks"]["mcmc_temperature"])

    def attack_factory(family, predictor, rng):
        common = {
            "protect_attachments": attack_cfg["protection"]["protect_attachments"],
            "protect_ring_closures": attack_cfg["protection"]["protect_ring_closures"],
            "protect_branches": attack_cfg["protection"]["protect_branches"],
        }
        if family == "randomization":
            return SmilesRandomizationAttack(rng, **common)
        if family == "probabilistic_mcmc":
            return ProbabilisticMCMCAttack(
                rng,
                predictor=predictor,
                allowed_tokens=vocab,
                steps=mcmc_steps,
                temperature=mcmc_temperature,
                **common,
            )
        if family == "substitution":
            cfg = attack_cfg["attacks"]["substitution"]
            return SubstitutionAttack(
                rng,
                allowed_tokens=vocab,
                attack_budget=cfg["attack_budget"],
                role_preserving=cfg["role_preserving"],
                **common,
            )
        if family == "insertion":
            cfg = attack_cfg["attacks"]["insertion"]
            return InsertionAttack(
                rng, allowed_tokens=vocab, attack_budget=cfg["attack_budget"], **common
            )
        if family == "deletion":
            cfg = attack_cfg["attacks"]["deletion"]
            return DeletionAttack(rng, attack_budget=cfg["attack_budget"], **common)
        if family == "rearrangement":
            cfg = attack_cfg["attacks"]["rearrangement"]
            return RearrangementAttack(rng, window_size=cfg["window"], **common)
        raise KeyError(f"unknown Phase 4 attack family: {family}")

    def build_augmented_dataset(defense, records, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(output_path)
        rows: list[dict] = []
        seen = set(clean_train[representation_column].astype(str))
        for record in records:
            if record.validity_status != "valid":
                continue
            representation = record.adversarial_representation
            if representation in seen:
                continue
            category = ATTACK_TAXONOMY[record.attack_type]
            if category is AttackCategory.REPRESENTATION_PRESERVING:
                if not canonical_graph_equivalent(
                    record.original_representation, record.adversarial_representation
                ):
                    raise ValueError(
                        "refusing measured-target inheritance without verified "
                        f"graph equivalence: {record.attack_id}"
                    )
                label = target_by_sample[record.sample_id]
                label_basis = "measured_target_from_graph_equivalent_source"
            else:
                label = record.adversarial_prediction
                label_basis = "clean_model_teacher_prediction_not_measured_target"
            if label is None:
                continue
            source = dataframe.iloc[int(record.sample_id.removeprefix("sample_"))].copy()
            source[representation_column] = representation
            source[target_column] = float(label)
            source["source_dataset"] = "phase4_adversarial_augmentation"
            source["phase4_attack_family"] = record.attack_type
            source["phase4_attack_category"] = category.value
            source["phase4_label_basis"] = label_basis
            rows.append(source.to_dict())
            seen.add(representation)
        clean_rows = clean_train.copy()
        clean_rows["phase4_attack_family"] = "clean"
        clean_rows["phase4_attack_category"] = "clean"
        clean_rows["phase4_label_basis"] = "measured_target"
        augmented = pd.concat([clean_rows, pd.DataFrame(rows)], ignore_index=True)
        augmented.to_csv(output_path, index=False)
        return output_path

    def train_defense(defense, augmented_path, model_dir, seed):
        from materials_adv.data.scaler import TargetScaler
        from materials_adv.models.regression import TransformerRegressor
        from materials_adv.training.train import train

        model = train(
            augmented_train_path=str(augmented_path),
            out_dir=str(model_dir),
            scaler_path=str(clean_model_dir / "scaler.json"),
            write_back_config=False,
            seed=seed,
        )
        scaler = TargetScaler.load(model_dir / "scaler.json")
        clean_scaler = TargetScaler.load(clean_model_dir / "scaler.json")
        if (scaler.mean, scaler.std) != (clean_scaler.mean, clean_scaler.std):
            raise ValueError(f"defense {defense.name} did not preserve the clean scaler")
        return TransformerRegressor(
            model, vocab, scaler, dataset_cfg["target_units"]
        )

    settings = ClosedLoopSettings(
        seed=int(phase4_cfg["seed"]),
        n_train_variants=int(phase4_cfg["attacks"]["training_variants"]),
        n_eval_variants=int(phase4_cfg["attacks"]["evaluation_variants"]),
        bootstrap_resamples=int(phase4_cfg["bootstrap"]["resamples"]),
        bootstrap_min_samples=int(phase4_cfg["bootstrap"]["min_source_samples"]),
        check_plausibility=True,
    )
    criterion = SuccessCriterion(**phase4_cfg["success_criterion"])
    defenses = [
        DefenseSpec(name, tuple(families))
        for name, families in phase4_cfg["defenses"].items()
    ]
    config_snapshot = {
        "phase4": phase4_cfg,
        "dataset": dataset_cfg,
        "model": model_cfg,
        "attack": attack_cfg,
        "tokenizer": tokenizer_cfg,
        "source": git_metadata(),
        "preflight": preflight,
    }
    input_artifacts = [
        Path(args.config),
        Path("configs/dataset.yaml"),
        Path("configs/model.yaml"),
        Path("configs/attack.yaml"),
        Path("configs/tokenizer.yaml"),
        processed_path,
        splits_path,
        vocab_path,
        clean_model_dir / "model.pt",
        clean_model_dir / "scaler.json",
        clean_model_dir / "metrics.json",
    ]
    experiment = ClosedLoopExperiment(
        args.run_dir,
        settings=settings,
        criterion=criterion,
        attack_families=phase4_cfg["attacks"]["families"],
        defenses=defenses,
        attack_factory=attack_factory,
    )
    experiment.run(
        clean_predictor=clean_predictor,
        clean_checkpoint=clean_model_dir / "model.pt",
        train_samples=train_samples,
        evaluation_examples=evaluation_examples,
        build_augmented_dataset=build_augmented_dataset,
        train_defended=train_defense,
        config_snapshot=config_snapshot,
        input_artifacts=input_artifacts,
    )
    print(f"Phase 4 artifacts written to {args.run_dir}")


if __name__ == "__main__":
    main()
