"""Reusable Phase 5 shortcut and representation-attribution experiment."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..attacks.randomization import SmilesRandomizationAttack
from ..data.tokenizer import tokenize
from ..evaluation.representation_attribution import (
    RidgeRegressor, length_features, occlusion_sensitivity, paired_bootstrap_ci,
    regression_metrics, representation_statistics, safe_correlation, token_features,
)


DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "HeavyAtomCount", "FractionCSP3",
)


@dataclass(frozen=True)
class AttributionSettings:
    seed: int = 20260910
    ridge_alpha: float = 1.0
    randomization_molecules: int = 100
    randomizations_per_molecule: int = 10
    permutation_molecules: int = 100
    occlusion_molecules: int = 25
    bootstrap_resamples: int = 2000
    bootstrap_minimum_n: int = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rdkit_descriptor_features(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    rows, valid = [], []
    for text in strings:
        mol = Chem.MolFromSmiles(text)
        valid.append(mol is not None)
        if mol is None:
            rows.append([np.nan] * len(DESCRIPTOR_NAMES))
            continue
        rows.append([
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol), Lipinski.RingCount(mol),
            Lipinski.HeavyAtomCount(mol), rdMolDescriptors.CalcFractionCSP3(mol),
        ])
    return np.asarray(rows, dtype=float), np.asarray(valid, dtype=bool)


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std


def _json_value(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


class RepresentationAttributionExperiment:
    def __init__(self, settings: AttributionSettings):
        self.settings = settings
        self.rng = np.random.default_rng(settings.seed)

    def run(
        self, *, train: pd.DataFrame, evaluation: pd.DataFrame,
        representation_column: str, target_column: str, vocab: Sequence[str],
        models: Mapping[str, object], checkpoint_paths: Mapping[str, Path],
        output_dir: Path, metadata: Mapping[str, object], config_snapshot: Mapping[str, object],
    ) -> dict:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite Phase 5 output: {output_dir}")
        output_dir.mkdir(parents=True)
        strings_train = train[representation_column].astype(str).tolist()
        strings_eval = evaluation[representation_column].astype(str).tolist()
        y_train = train[target_column].to_numpy(float)
        y_eval = evaluation[target_column].to_numpy(float)

        baseline_predictions: dict[str, np.ndarray] = {}
        baseline_predictions["mean"] = np.full(len(y_eval), y_train.mean())
        length_model = RidgeRegressor(self.settings.ridge_alpha).fit(length_features(strings_train), y_train)
        baseline_predictions["length_only"] = length_model.predict(length_features(strings_eval))
        for name, frequencies in (("bag_of_token_counts", False), ("bag_of_token_frequencies", True)):
            x_train = token_features(strings_train, vocab, frequencies=frequencies)
            x_eval = token_features(strings_eval, vocab, frequencies=frequencies)
            baseline_predictions[name] = RidgeRegressor(self.settings.ridge_alpha).fit(x_train, y_train).predict(x_eval)

        desc_train, valid_train = rdkit_descriptor_features(strings_train)
        desc_eval, valid_eval = rdkit_descriptor_features(strings_eval)
        x_train, x_eval = _standardize(desc_train[valid_train], desc_eval[valid_eval])
        descriptor_prediction = np.full(len(y_eval), np.nan)
        descriptor_prediction[valid_eval] = RidgeRegressor(self.settings.ridge_alpha).fit(
            x_train, y_train[valid_train]
        ).predict(x_eval)
        baseline_predictions["rdkit_descriptors"] = descriptor_prediction
        for name, model in models.items():
            baseline_predictions[name] = model.predict(strings_eval)

        metrics = {}
        prediction_rows = evaluation[[representation_column, target_column]].copy()
        for name, predictions in baseline_predictions.items():
            keep = np.isfinite(predictions)
            metrics[name] = {**regression_metrics(y_eval[keep], predictions[keep]),
                             "n": int(keep.sum()), "coverage": float(keep.mean())}
            prediction_rows[name] = predictions
        prediction_rows.to_csv(output_dir / "baseline_predictions.csv", index=False)

        # Composition-preserving but generally invalid token-order probes.
        permutation_rows = []
        permutation_examples = evaluation.sample(
            n=min(self.settings.permutation_molecules, len(evaluation)),
            random_state=self.settings.seed,
        )
        for row_index, text in zip(permutation_examples.index, permutation_examples[representation_column].astype(str)):
            tokens = tokenize(text)
            shuffled = list(self.rng.permutation(tokens))
            reversed_tokens = tokens[::-1]
            for model_name, model in models.items():
                values = model.predict([text, "".join(shuffled), "".join(reversed_tokens)])
                permutation_rows.append({"row_index": row_index, "model": model_name,
                    "token_length": len(tokens), "clean_prediction": values[0],
                    "shuffle_prediction": values[1], "reverse_prediction": values[2],
                    "shuffle_absolute_drift": abs(values[1] - values[0]),
                    "reverse_absolute_drift": abs(values[2] - values[0]),
                    "chemical_status": "not_evaluated_nonchemical_positional_probe"})
        permutation_df = pd.DataFrame(permutation_rows)
        permutation_df.to_csv(output_dir / "permutation_sensitivity.csv", index=False)

        # True graph-equivalent representation controls. Reuse the same candidate bank for every model.
        randomized_rows, molecule_rows = [], []
        from rdkit import Chem
        eval_random = evaluation.sample(
            n=min(self.settings.randomization_molecules, len(evaluation)),
            random_state=self.settings.seed + 1,
        )
        reference_scale = max(metrics["clean_transformer"]["mae"], np.finfo(float).eps)
        model_scores: dict[str, list[float]] = {name: [] for name in models}
        for row_index, text in zip(eval_random.index, eval_random[representation_column].astype(str)):
            attack = SmilesRandomizationAttack(self.rng)
            outcomes = attack.generate(tokenize(text), self.settings.randomizations_per_molecule)
            representations = [text] + ["".join(outcome.adversarial_tokens) for outcome in outcomes]
            if len(representations) < 2:
                continue
            lengths = [len(tokenize(value)) for value in representations]
            original_mol = Chem.MolFromSmiles(text)
            original_canonical = Chem.MolToSmiles(original_mol, canonical=True) if original_mol else None
            per_model = {}
            for model_name, model in models.items():
                predictions = model.predict(representations)
                embeddings = model.encode(representations)
                stats = representation_statistics(predictions, lengths, embeddings,
                                                  reference_scale=reference_scale)
                per_model[model_name] = stats
                model_scores[model_name].append(float(stats["representation_invariance_score"]))
                molecule_rows.append({"row_index": row_index, "model": model_name, **stats})
                for rep_index, (representation, length, prediction) in enumerate(zip(representations, lengths, predictions)):
                    candidate_mol = Chem.MolFromSmiles(representation)
                    candidate_canonical = (Chem.MolToSmiles(candidate_mol, canonical=True)
                                           if candidate_mol else None)
                    randomized_rows.append({"row_index": row_index, "model": model_name,
                        "representation_index": rep_index, "is_original": rep_index == 0,
                        "representation": representation, "token_length": length,
                        "prediction": prediction,
                        "validity_status": "valid" if candidate_mol else "invalid",
                        "canonical_equivalent": bool(original_canonical is not None and
                                                     candidate_canonical == original_canonical)})
        randomized_df, molecule_df = pd.DataFrame(randomized_rows), pd.DataFrame(molecule_rows)
        randomized_df.to_csv(output_dir / "randomized_representation_predictions.csv", index=False)
        molecule_df.to_csv(output_dir / "per_molecule_invariance.csv", index=False)

        transfer = {}
        if "clean_transformer" in model_scores and "defended_transformer" in model_scores:
            delta = np.asarray(model_scores["defended_transformer"]) - np.asarray(model_scores["clean_transformer"])
            transfer = paired_bootstrap_ci(delta, rng=self.rng,
                n_resamples=self.settings.bootstrap_resamples,
                minimum_n=self.settings.bootstrap_minimum_n)

        occlusion_rows = []
        occlusion_examples = evaluation.sample(
            n=min(self.settings.occlusion_molecules, len(evaluation)),
            random_state=self.settings.seed + 2,
        )
        for row_index, text in zip(occlusion_examples.index,
                                   occlusion_examples[representation_column].astype(str)):
            tokens = tokenize(text)
            for model_name, model in models.items():
                for item in occlusion_sensitivity(tokens, model.predict_token_lists):
                    occlusion_rows.append({"row_index": row_index, "model": model_name,
                        "probe_status": "token_deletion_may_be_invalid_smiles", **item})
        pd.DataFrame(occlusion_rows).to_csv(output_dir / "token_occlusion.csv", index=False)

        positional_summary = {}
        for name, group in permutation_df.groupby("model"):
            positional_summary[name] = {
                "n": len(group), "mean_shuffle_drift": float(group.shuffle_absolute_drift.mean()),
                "median_shuffle_drift": float(group.shuffle_absolute_drift.median()),
                "mean_reverse_drift": float(group.reverse_absolute_drift.mean()),
                "median_reverse_drift": float(group.reverse_absolute_drift.median()),
            }
        invariance_summary = {name: {"n": len(values), "mean_score": float(np.mean(values)),
            "median_score": float(np.median(values))} for name, values in model_scores.items() if values}
        summary = {
            "status": "observed", "target_units": metadata.get("target_units"),
            "baseline_metrics": metrics,
            "dataset_dependence": {"length_target_correlation_train": safe_correlation(
                length_features(strings_train).ravel(), y_train)},
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "descriptor_validity": {"train_valid": int(valid_train.sum()), "train_total": len(valid_train),
                "evaluation_valid": int(valid_eval.sum()), "evaluation_total": len(valid_eval)},
            "positional_sensitivity": positional_summary,
            "representation_invariance": invariance_summary,
            "randomization_validation": {
                "records": int(len(randomized_df)),
                "validity_rate": float((randomized_df.validity_status == "valid").mean()),
                "canonical_equivalence_rate": float(randomized_df.canonical_equivalent.mean()),
            },
            "defended_minus_clean_invariance_score": transfer,
            "interpretation_guardrails": {
                "randomization": "RDKit graph-equivalent label-preserving control",
                "permutation": "composition-preserving positional probe; chemical validity not assumed",
                "occlusion": "model sensitivity probe; chemical validity and target preservation not assumed",
                "attention": "not used as a standalone explanation",
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_value) + "\n")
        import rdkit, torch
        reproducibility = {**metadata, "settings": self.settings.__dict__,
            "checkpoint_hashes": {name: sha256_file(path) for name, path in checkpoint_paths.items()},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "torch": torch.__version__,
                        "rdkit": rdkit.__version__}}
        (output_dir / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2) + "\n")
        (output_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2) + "\n")
        return summary
