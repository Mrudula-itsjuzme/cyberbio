"""Run the validation-only representation-robustness review.

This script does not train or tune a model and never opens the sealed test split.
It evaluates the frozen canonical Transformer, optionally evaluates existing
augmentation-trained and graph checkpoints, and writes machine-readable results.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.evaluation.integrity import audit_active_lineage, sha256
from materials_adv.evaluation.multiview import (
    disagreement_bins,
    evaluate_multiview,
    molecule_bootstrap_mae_delta,
    summarize_multiview,
)
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.seeding import seed_everything


class SequencePredictor:
    def __init__(self, model, vocab: list[str], scaler: TargetScaler, device: torch.device):
        self.model, self.scaler, self.device = model.eval(), scaler, device
        self.lookup = {token: i + 1 for i, token in enumerate(vocab)}

    def __call__(self, smiles: list[str]) -> np.ndarray:
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(smiles), 64):
                batch = smiles[start:start + 64]
                encoded = []
                for value in batch:
                    ids = [self.lookup.get(t, 0) for t in tokenize(value)][:256]
                    encoded.append(ids + [0] * (256 - len(ids)))
                x = torch.tensor(encoded, dtype=torch.long, device=self.device)
                out = self.model(x, padding_mask=x.eq(0)).detach().cpu().numpy()
                predictions.extend(np.asarray(self.scaler.inverse_transform(out)).reshape(-1))
        return np.asarray(predictions, dtype=float)


class GraphModelPredictor:
    def __init__(self, model, scaler: TargetScaler, device: torch.device):
        self.model, self.scaler, self.device = model.eval(), scaler, device

    def __call__(self, smiles: list[str]) -> np.ndarray:
        frame = pd.DataFrame({"original_representation": smiles, "property_value": np.zeros(len(smiles))})
        dataset = GraphDataset(frame, strict=True)
        predictions = []
        with torch.inference_mode():
            for item in dataset:
                out = self.model(
                    item["x"].unsqueeze(0).to(self.device),
                    item["adj"].unsqueeze(0).to(self.device),
                    item["mask"].unsqueeze(0).to(self.device),
                ).detach().cpu().numpy()
                predictions.extend(np.asarray(self.scaler.inverse_transform(out)).reshape(-1))
        return np.asarray(predictions, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--n-views", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--limit", type=int, default=None, help="Validation-only smoke subset")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    seed_record = seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir or root / "results" / "representation_robustness_review" / str(int(time.time()))
    out.mkdir(parents=True, exist_ok=False)

    checkpoint = root / "results/models/transformer_regressor/model.pt"
    scaler_path = root / "results/models/transformer_regressor/scaler.json"
    integrity = audit_active_lineage(root, checkpoint, scaler_path)
    (out / "integrity_audit.json").write_text(json.dumps(integrity, indent=2) + "\n")

    frame = pd.read_csv(root / "data/processed/processed.csv")
    splits = json.loads((root / "data/processed/splits.json").read_text())
    validation = frame.iloc[splits["val"]].copy().reset_index(drop=True)
    if args.limit is not None:
        validation = validation.iloc[:args.limit].copy()
    vocab = json.loads((root / "data/processed/vocab.json").read_text())
    scaler = TargetScaler.load(scaler_path)

    ordinary = TransformerRegressorModel(
        vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4,
        dim_feedforward=128, dropout=0.1, max_seq_len=256, pooling="mean",
    ).to(device)
    ordinary.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model_specs = {
        "transformer_canonical": (SequencePredictor(ordinary, vocab, scaler, device), checkpoint, "frozen canonical checkpoint"),
    }

    historical = root / "results/phase11c_canonical_verification/1789326819/models"
    augmented_path = historical / "transformer_augmented/model.pt"
    if augmented_path.exists():
        augmented = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
        augmented.load_state_dict(torch.load(augmented_path, map_location=device, weights_only=True))
        model_specs["transformer_augmented_existing"] = (
            SequencePredictor(augmented, vocab, scaler, device), augmented_path,
            "existing checkpoint; training seed/provenance incomplete, evaluated without retraining",
        )
    graph_path = historical / "graph_mpnn_small/model.pt"
    if graph_path.exists():
        graph = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(device)
        graph.load_state_dict(torch.load(graph_path, map_location=device, weights_only=True))
        model_specs["graph_mpnn_existing"] = (
            GraphModelPredictor(graph, scaler, device), graph_path,
            "existing checkpoint; training seed/provenance incomplete, graph invariance control only",
        )

    summaries = {}
    for name, (predictor, path, limitation) in model_specs.items():
        started = time.perf_counter()
        per_molecule = evaluate_multiview(
            validation, predictor, n_views=args.n_views, seed=args.seed
        )
        elapsed = time.perf_counter() - started
        per_molecule.to_csv(out / f"{name}_per_molecule.csv", index=False)
        disagreement_bins(per_molecule).to_csv(out / f"{name}_disagreement_bins.csv", index=False)
        summary = summarize_multiview(per_molecule)
        summary["paired_bootstrap_mean_vs_canonical"] = molecule_bootstrap_mae_delta(
            per_molecule, seed=args.seed
        )
        summary["runtime_seconds"] = elapsed
        summary["prediction_evaluations"] = int(
            len(per_molecule) * 2 + per_molecule["unique_views"].sum()
        )
        summary["checkpoint_sha256"] = sha256(path)
        summary["limitation"] = limitation
        summaries[name] = summary

    reproducibility = {
        "status": "completed",
        "evaluation_split": "validation",
        "sealed_test_opened": False,
        "seed": args.seed,
        "seed_record": seed_record,
        "requested_views_per_molecule": args.n_views,
        "n_molecules": len(validation),
        "models": list(model_specs),
        "summaries": summaries,
    }
    (out / "summary.json").write_text(json.dumps(reproducibility, indent=2, allow_nan=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
