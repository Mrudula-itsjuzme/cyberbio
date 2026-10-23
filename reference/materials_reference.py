#!/usr/bin/env python3
"""Single-file reference façade for the modular Materials project.

This file is intentionally an orchestration/reference layer, not a copy of the
implementation.  The maintained code remains in ``materials-adversarial/src``.

Pipeline
--------
raw polymer data -> audit/split -> PSMILES tokenization -> Transformer/GraphMPNN
-> clean evaluation -> representation controls/chemistry stress tests -> validity
and budget accounting -> optional physical oracle -> reports in ``results/materials``

Examples (from repository root)
--------------------------------
python reference/materials_reference.py map
python reference/materials_reference.py tokenize "[*]CC(=O)O[*]"
python reference/materials_reference.py metrics --truth 1,2,3 --prediction 1.1,1.8,3.2
python reference/materials_reference.py results

RDKit/PyTorch-dependent commands require the Materials environment described in
``materials-adversarial/pyproject.toml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "materials-adversarial"
SOURCE_ROOT = PROJECT_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results" / "materials"

MODULE_MAP = {
    "data": "materials_adv.data: loading, deterministic splits, scaling, tokenization",
    "models": "materials_adv.models: Transformer, specialized Transformer, GraphMPNN",
    "attacks": "materials_adv.attacks: controls, edit operators, registries, attack API",
    "framework": "materials_adv.framework: budgets, objectives, predictors, search contracts",
    "validation": "materials_adv.validation: chemistry, representation, and sanity gates",
    "evaluation": "materials_adv.evaluation: clean, robustness, integrity, multiview metrics",
    "experiments": "materials_adv.experiments: paired, efficiency, attribution, closed-loop runs",
    "oracle": "materials_adv.oracle + hpc_oracle: candidate selection and independent evaluation",
    "training": "materials_adv.training: training and held-out evaluation",
}


def _enable_project_imports() -> None:
    """Add the maintained package to sys.path only when an API command needs it."""
    source = str(SOURCE_ROOT)
    project = str(PROJECT_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    if project not in sys.path:
        sys.path.insert(0, project)


def tokenize_psmiles(text: str) -> list[str]:
    """Tokenize PSMILES using the canonical chemistry-aware tokenizer."""
    _enable_project_imports()
    from materials_adv.data.tokenizer import tokenize

    return tokenize(text)


def regression_summary(truth: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    """Compute canonical MAE, RMSE, and R-squared metrics."""
    _enable_project_imports()
    from materials_adv.evaluation.metrics import regression_metrics

    import numpy as np

    return regression_metrics(np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float))


def build_transformer(*, vocab_size: int, max_seq_len: int = 256):
    """Construct the small canonical reference Transformer; training is separate."""
    _enable_project_imports()
    from materials_adv.models.transformer import TransformerRegressorModel

    return TransformerRegressorModel(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=max_seq_len,
        pooling="mean",
    )


def result_inventory() -> dict[str, int]:
    """Count preserved result artifacts by top-level result group."""
    if not RESULTS_ROOT.exists():
        return {}
    return {
        path.name: sum(1 for item in path.rglob("*") if item.is_file())
        for path in sorted(RESULTS_ROOT.iterdir())
        if path.is_dir()
    }


def _csv_numbers(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("map", help="show the modular implementation map")
    tok = sub.add_parser("tokenize", help="tokenize one PSMILES string")
    tok.add_argument("psmiles")
    metric = sub.add_parser("metrics", help="compute regression metrics")
    metric.add_argument("--truth", required=True, type=_csv_numbers)
    metric.add_argument("--prediction", required=True, type=_csv_numbers)
    sub.add_parser("results", help="summarize consolidated result artifacts")
    args = parser.parse_args()

    if args.command == "map":
        print(json.dumps(MODULE_MAP, indent=2))
    elif args.command == "tokenize":
        print(json.dumps(tokenize_psmiles(args.psmiles)))
    elif args.command == "metrics":
        print(json.dumps(regression_summary(args.truth, args.prediction), indent=2))
    elif args.command == "results":
        print(json.dumps(result_inventory(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
