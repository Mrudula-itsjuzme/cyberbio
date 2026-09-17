"""Framework-V2 adapters that bind real property predictors to the generic
``AttackObjective`` / ``SearchStrategy`` interfaces.

Forensic note (see docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md): the original
``scripts/benchmark_v2.py`` never used a real model.  It scored candidates with
``float(hash(smiles) % 1000) / 100.0``, a process-local Python hash stub whose
values are uniform in [0, 10).  Any "drift" reported by that script is therefore
independent of chemistry and of the canonical GraphMPNN.

This module supplies the predictor that the framework was missing so that
benchmarks can be run against the canonical model with a single, auditable
inverse-scaling step (``abs(candidate_eV - source_eV)``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.data.scaler import TargetScaler
from materials_adv.framework.interfaces import Candidate, Predictor
from materials_adv.models.graph_predictor import GraphPredictor

DEFAULT_CHECKPOINT = (
    "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
)
DEFAULT_SCALER = "results/models/transformer_regressor/scaler.json"


def _as_smiles(representation: Any) -> str:
    if isinstance(representation, Candidate):
        return representation.identifier
    return str(representation)


class GraphMPNNPredictor(Predictor):
    """Canonical GraphMPNN (``GraphPredictor(node_dim=7, hidden_dim=64,
    num_layers=3)``) exposed through the framework ``Predictor`` interface.

    ``predict`` returns a bandgap in eV: the model emits a scaled value which is
    inverse transformed exactly once with the canonical target scaler.
    """

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        scaler_path: str | Path = DEFAULT_SCALER,
        device: str | torch.device | None = None,
        node_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 3,
        batch_size: int = 32,
        strict: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.scaler_path = Path(scaler_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        # Adversarial evaluation must never score a substituted molecule.
        self.strict = strict

        self.model = GraphPredictor(node_dim=node_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.scaler = TargetScaler.load(self.scaler_path)

        # Accounting for query-budget audits: every scored molecule is counted here.
        self.forward_calls = 0
        self.molecules_scored = 0
        self.silent_substitutions = 0   # always 0 when strict

    # -- core inference ---------------------------------------------------
    def predict_batch(self, representations: Iterable[Any]) -> List[float]:
        smiles = [_as_smiles(r) for r in representations]
        if not smiles:
            return []

        # NOTE: no RDKit import in the framework layer. Parse failures are surfaced by
        # GraphDataset(strict=True) instead of being probed here.
        df = pd.DataFrame({"original_representation": smiles, "property_value": [0.0] * len(smiles)})
        dataset = GraphDataset(df, strict=getattr(self, "strict", True))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        out: List[float] = []
        with torch.no_grad():
            for batch in loader:
                y_hat = self.model(
                    batch["x"].to(self.device),
                    batch["adj"].to(self.device),
                    batch["mask"].to(self.device),
                )
                self.forward_calls += 1
                scaled = y_hat.view(-1).cpu().numpy().reshape(-1, 1)
                out.extend(self.scaler.inverse_transform(scaled).flatten().tolist())
        self.molecules_scored += len(smiles)
        self.silent_substitutions += getattr(dataset, "skipped_unparseable", 0)
        return out

    def predict(self, representation: Any) -> float:
        return self.predict_batch([representation])[0]

    # -- introspection ----------------------------------------------------
    def checkpoint_sha256(self) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(self.checkpoint, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def config(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256(),
            "scaler": json.loads(Path(self.scaler_path).read_text()),
            "node_dim": 7,
            "hidden_dim": 64,
            "num_layers": 3,
            "parameters": self.parameter_count(),
            "strict": self.strict,
        }

    def validate_against_release_manifest(self, manifest_path: str | Path) -> None:
        """Fail loudly if the checkpoint/scaler do not match the frozen release."""
        manifest = json.loads(Path(manifest_path).read_text())
        expected_ckpt = manifest["hashes"]["GraphMPNN_Small_ckpt"]
        expected_scaler = manifest["hashes"]["scaler"]
        if self.checkpoint_sha256() != expected_ckpt:
            raise ValueError(
                f"GraphMPNN checkpoint hash {self.checkpoint_sha256()} != release manifest {expected_ckpt}"
            )
        import hashlib

        scaler_hash = hashlib.sha256(Path(self.scaler_path).read_bytes()).hexdigest()
        if scaler_hash != expected_scaler:
            raise ValueError(
                f"target scaler hash {scaler_hash} != release manifest {expected_scaler}"
            )
        if self.parameter_count() != manifest.get("parameter_counts", {}).get(
            "GraphMPNN_Small", 27585
        ):
            raise ValueError("GraphMPNN parameter count does not match the release manifest")


class HashStubPredictor(Predictor):
    """MOCK_TEST_ONLY -- reproduction of the placeholder predictor used by the
    developmental Framework-V2 benchmark: ``float(hash(smiles) % 1000) / 100.0``.

    It exists only so the forensic audit can quantify the artefact and so tests can
    assert that benchmarks never fall back to it. It must never be used to produce a
    reported number. Declared namespace: MOCK_TEST_ONLY.
    """

    def predict(self, representation: Any) -> float:
        return float(hash(_as_smiles(representation)) % 1000) / 100.0

    def predict_batch(self, representations: Iterable[Any]) -> List[float]:
        return [self.predict(r) for r in representations]
