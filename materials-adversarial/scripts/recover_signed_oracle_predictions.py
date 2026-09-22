"""Recover signed frozen-model predictions for preregistered oracle pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(smiles, vocab, max_len=256):
    mapping = {token: index + 1 for index, token in enumerate(vocab)}
    ids, masks = [], []
    for value in smiles:
        row = [mapping.get(token, 0) for token in tokenize(value)][:max_len]
        mask = [False] * len(row)
        row += [0] * (max_len - len(row))
        mask += [True] * (max_len - len(mask))
        ids.append(row)
        masks.append(mask)
    return torch.tensor(ids), torch.tensor(masks, dtype=torch.bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = torch.device("cpu")

    graph_path = ROOT / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
    transformer_path = ROOT / "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"
    scaler_path = ROOT / "results/models/transformer_regressor/scaler.json"
    vocab_path = ROOT / "data/processed/vocab.json"
    for path in (graph_path, transformer_path, scaler_path, vocab_path, args.pairs):
        if not path.is_file():
            raise FileNotFoundError(path)

    scaler = TargetScaler.load(scaler_path)
    vocab = json.loads(vocab_path.read_text())
    graph = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(device)
    graph.load_state_dict(torch.load(graph_path, map_location=device, weights_only=True))
    graph.eval()
    transformer = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
    transformer.load_state_dict(torch.load(transformer_path, map_location=device, weights_only=True))
    transformer.eval()

    def predict_graph(values):
        dataset = GraphDataset(pd.DataFrame({"original_representation": values,
                                             "property_value": [0.0] * len(values)}))
        result = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=32, shuffle=False):
                raw = graph(batch["x"], batch["adj"], batch["mask"])
                result.extend(scaler.inverse_transform(raw.reshape(-1, 1).numpy()).ravel())
        return result

    def predict_transformer(values):
        token_ids, padding_mask = encode(values, vocab)
        with torch.no_grad():
            raw = transformer(token_ids, padding_mask=padding_mask)
        return scaler.inverse_transform(raw.reshape(-1, 1).numpy()).ravel()

    pairs = pd.read_csv(args.pairs)
    rows = []
    for row in pairs.to_dict("records"):
        values = [row["source_smiles"], row["candidate_smiles"]]
        fn = predict_graph if row["target_model"] == "GraphMPNN" else predict_transformer
        source, candidate = map(float, fn(values))
        signed = candidate - source
        if not np.isclose(abs(signed), float(row["model_drift_eV"]), atol=2e-5, rtol=0):
            raise RuntimeError(f"Drift mismatch for source {row['source_id']}: {signed}")
        rows.append({
            "source_id": int(row["source_id"]), "attack_family": row["attack_family"],
            "target_model": row["target_model"], "source_smiles": row["source_smiles"],
            "candidate_smiles": row["candidate_smiles"],
            "source_prediction_eV": source, "candidate_prediction_eV": candidate,
            "delta_G_signed_eV": signed, "absolute_drift_eV": abs(signed),
            "record_status": "MODEL_ONLY_NO_ORACLE_RESULT",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "signed_model_predictions.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    manifest = {
        "status": "complete_model_predictions_only",
        "sealed_test_accessed": False,
        "device": "cpu", "torch_deterministic_algorithms": True, "torch_seed": 0,
        "definition": "delta_G_signed_eV = G(candidate) - G(source)",
        "artifacts": {
            "graph_checkpoint": {"path": str(graph_path.relative_to(ROOT)), "sha256": sha256(graph_path)},
            "transformer_checkpoint": {"path": str(transformer_path.relative_to(ROOT)), "sha256": sha256(transformer_path)},
            "scaler": {"path": str(scaler_path.relative_to(ROOT)), "sha256": sha256(scaler_path)},
            "vocabulary": {"path": str(vocab_path.relative_to(ROOT)), "sha256": sha256(vocab_path)},
            "pair_manifest": {"path": str(args.pairs), "sha256": sha256(args.pairs)},
            "predictions": {"path": str(output), "sha256": sha256(output)},
        },
    }
    (args.output_dir / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
