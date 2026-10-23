#!/usr/bin/env python3
"""Single-file reference façade for the modular synthetic Bio-Cyber project.

This project uses synthetic DNA-like sequences only.  The maintained modules are
in ``bio-cyber-adversarial/src``; this file provides one readable entrypoint.

Pipeline
--------
deterministic synthetic generation -> stratified splits -> one-hot encoding ->
MotifCNN training -> held-out metrics/interpretability -> motif-protected attack
-> defense evaluation -> reports in ``results/bio-cyber``

Examples (from repository root)
--------------------------------
python reference/bio_reference.py map
python reference/bio_reference.py generate --samples 20 --output /tmp/bio-smoke.csv
python reference/bio_reference.py encode ACGTACGT
python reference/bio_reference.py model --sequence-length 50
python reference/bio_reference.py results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "bio-cyber-adversarial"
SOURCE_ROOT = PROJECT_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results" / "bio-cyber"

MODULE_MAP = {
    "data_generator": "balanced deterministic synthetic motif dataset",
    "dataset/encoding": "split-aware dataset loading and one-hot encoding",
    "model": "small one-dimensional MotifCNN classifier",
    "train/evaluate": "baseline fitting and held-out classification metrics",
    "attack": "gradient-guided substitutions outside the implanted motif",
    "train_defended/evaluate_defended": "adversarial defense and transfer evaluation",
}


def _enable_project_imports() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def generate_synthetic_dataset(samples: int, output: Path, seed: int = 42) -> Path:
    """Generate the canonical safe synthetic dataset at an explicit path."""
    _enable_project_imports()
    from data_generator import generate_dataset

    output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(num_samples=samples, seed=seed, output_path=str(output))
    return output


def encode_sequence(sequence: str):
    """Return the canonical four-channel one-hot tensor for a sequence."""
    _enable_project_imports()
    from encoding import one_hot_encode

    return one_hot_encode(sequence)


def build_model(sequence_length: int = 50):
    """Construct the canonical MotifCNN without loading or training weights."""
    _enable_project_imports()
    from model import MotifCNN

    return MotifCNN(sequence_length=sequence_length)


def result_inventory() -> dict[str, int]:
    if not RESULTS_ROOT.exists():
        return {}
    return {
        path.name: sum(1 for item in path.rglob("*") if item.is_file())
        for path in sorted(RESULTS_ROOT.iterdir())
        if path.is_dir()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("map", help="show the modular implementation map")
    generate = sub.add_parser("generate", help="generate a small synthetic dataset")
    generate.add_argument("--samples", type=int, default=20)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path, required=True)
    encode = sub.add_parser("encode", help="one-hot encode one synthetic sequence")
    encode.add_argument("sequence")
    model = sub.add_parser("model", help="print the canonical CNN architecture")
    model.add_argument("--sequence-length", type=int, default=50)
    sub.add_parser("results", help="summarize consolidated result artifacts")
    args = parser.parse_args()

    if args.command == "map":
        print(json.dumps(MODULE_MAP, indent=2))
    elif args.command == "generate":
        print(generate_synthetic_dataset(args.samples, args.output, args.seed))
    elif args.command == "encode":
        encoded = encode_sequence(args.sequence)
        print(json.dumps({"shape": list(encoded.shape), "values": encoded.tolist()}))
    elif args.command == "model":
        print(build_model(args.sequence_length))
    elif args.command == "results":
        print(json.dumps(result_inventory(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
