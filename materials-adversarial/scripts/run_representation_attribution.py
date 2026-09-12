#!/usr/bin/env python3
"""Run Phase 5 attribution without retraining or overwriting prior outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from materials_adv.data.scaler import TargetScaler
from materials_adv.experiments.representation_attribution import (
    AttributionSettings, RepresentationAttributionExperiment,
)
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/representation_attribution.yaml")
    parser.add_argument("--output-dir", default="results/representation_attribution")
    return parser.parse_args()


def load_predictor(path: Path, model_config: dict, vocab: list[str], units: str):
    import torch
    architecture = model_config["architecture"]
    model = TransformerRegressorModel(
        vocab_size=len(vocab), d_model=architecture["d_model"],
        n_layers=architecture["n_layers"], n_heads=architecture["n_heads"],
        dim_feedforward=architecture["dim_feedforward"], dropout=architecture["dropout"],
        max_seq_len=architecture["max_seq_len"], pooling=architecture["pooling"],
    )
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return TransformerRegressor(model, vocab, TargetScaler.load(path / "scaler.json"), units)


def git_metadata() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], check=True,
                                capture_output=True, text=True).stdout)
    return {"git_commit": commit, "working_tree_dirty": dirty}


def main() -> None:
    args = arguments()
    phase = yaml.safe_load(Path(args.config).read_text())
    dataset_config, model_config = load_config("dataset"), load_config("model")
    if phase["lineage"] != dataset_config["name"]:
        raise ValueError("Phase 5 and active dataset lineages differ")
    processed = Path(dataset_config["processed_dir"])
    dataframe = pd.read_csv(processed / "processed.csv")
    splits = json.loads((processed / "splits.json").read_text())
    vocab = json.loads((processed / "vocab.json").read_text())
    model_paths = {name: Path(value) for name, value in phase["models"].items()}
    models = {name: load_predictor(path, model_config, vocab, dataset_config["target_units"])
              for name, path in model_paths.items()}
    settings = AttributionSettings(**{key: phase[key] for key in AttributionSettings.__dataclass_fields__})
    experiment = RepresentationAttributionExperiment(settings)
    summary = experiment.run(
        train=dataframe.iloc[splits["train"]].copy(),
        evaluation=dataframe.iloc[splits[phase["evaluation_split"]]].copy(),
        representation_column="original_representation", target_column="property_value",
        vocab=vocab, models=models,
        checkpoint_paths={name: path / "model.pt" for name, path in model_paths.items()},
        output_dir=Path(args.output_dir),
        metadata={**git_metadata(), "lineage": phase["lineage"],
                  "evaluation_split": phase["evaluation_split"],
                  "target_units": dataset_config["target_units"],
                  "processed_data": str(processed / "processed.csv")},
        config_snapshot={"phase5": phase, "dataset": dataset_config, "model": model_config},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
