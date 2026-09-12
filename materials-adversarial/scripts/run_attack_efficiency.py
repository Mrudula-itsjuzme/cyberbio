#!/usr/bin/env python3
"""Run the Phase 6 equal-query-budget black-box attack benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.search import CompositeProposalOperator
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.experiments.attack_efficiency import (
    AttackEfficiencyBenchmark, AttackEfficiencySettings,
)
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/attack_efficiency.yaml")
    parser.add_argument("--output-dir", default="results/attack_efficiency")
    return parser.parse_args()


def load_predictor(path: Path, config: dict, vocab: list[str], units: str):
    import torch
    architecture = config["architecture"]
    model = TransformerRegressorModel(len(vocab), architecture["d_model"],
        architecture["n_layers"], architecture["n_heads"],
        architecture["dim_feedforward"], architecture["dropout"],
        architecture["max_seq_len"], architecture["pooling"])
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return TransformerRegressor(model, vocab, TargetScaler.load(path / "scaler.json"), units)


def main():
    args = arguments()
    phase = yaml.safe_load(Path(args.config).read_text())
    dataset_config, model_config, attack_config = (
        load_config("dataset"), load_config("model"), load_config("attack"))
    if phase["lineage"] != dataset_config["name"]:
        raise ValueError("Phase 6 and active dataset lineages differ")
    processed = Path(dataset_config["processed_dir"])
    dataframe = pd.read_csv(processed / "processed.csv")
    splits = json.loads((processed / "splits.json").read_text())
    vocab = json.loads((processed / "vocab.json").read_text())
    candidates = dataframe.iloc[splits[phase["evaluation_split"]]].sample(
        n=min(phase["n_examples"], len(splits[phase["evaluation_split"]])),
        random_state=phase["seed"])
    samples = [(f"row_{index}", str(row.original_representation))
               for index, row in candidates.iterrows()]
    model_dir = Path(phase["model_dir"])
    predictor = load_predictor(model_dir, model_config, vocab, dataset_config["target_units"])
    protection = attack_config["protection"]

    def proposal_factory(rng):
        operators = []
        for name in phase["proposal_suite"]["operators"]:
            if name == "substitution":
                operators.append(SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1,
                    role_preserving=phase["proposal_suite"]["role_preserving_substitution"], **protection))
            elif name == "insertion":
                operators.append(InsertionAttack(rng, allowed_tokens=vocab, attack_budget=1, **protection))
            elif name == "deletion":
                operators.append(DeletionAttack(rng, attack_budget=1, **protection))
            elif name == "rearrangement":
                operators.append(RearrangementAttack(rng,
                    window_size=phase["proposal_suite"]["rearrangement_window"], **protection))
            else:
                raise KeyError(f"unsupported proposal operator {name!r}")
        return CompositeProposalOperator(operators, rng)

    fields = AttackEfficiencySettings.__dataclass_fields__
    settings = AttackEfficiencySettings(**{name: phase[name] for name in fields})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], check=True,
                                capture_output=True, text=True).stdout)
    summary = AttackEfficiencyBenchmark(settings).run(
        samples=samples, predictor=predictor, proposal_factory=proposal_factory,
        output_dir=Path(args.output_dir), checkpoint_path=model_dir / "model.pt",
        metadata={"git_commit": commit, "working_tree_dirty": dirty,
                  "lineage": phase["lineage"], "evaluation_split": phase["evaluation_split"],
                  "target_units": dataset_config["target_units"]},
        config_snapshot={"phase6": phase, "dataset": dataset_config,
                         "model": model_config, "attack": attack_config})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
