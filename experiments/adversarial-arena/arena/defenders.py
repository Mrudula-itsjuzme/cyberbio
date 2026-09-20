from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np

from .contracts import CandidateRecord, Defender, SourceSample


class RidgeSmokeDefender(Defender):
    """Deterministic, real-data smoke defender; not a replacement for TransPolymer.

    Uses hashed character n-grams so a two-round loop can be proven cheaply. Only
    label-preserving RandomSMILES hard examples receive their source labels.
    """

    def __init__(self, seed: int, dimension: int = 256, regularization: float = 1.0):
        self.seed, self.dimension, self.regularization = seed, dimension, regularization
        self.round_id, self.checkpoint_id = -1, "untrained"
        self.weights = np.zeros(dimension + 1)
        self._metrics: dict = {}

    def _features(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension)
        for n in (1, 2, 3):
            for i in range(max(0, len(text) - n + 1)):
                token = text[i:i+n]
                index = int.from_bytes(__import__("hashlib").sha256(token.encode()).digest()[:8], "big") % self.dimension
                vec[index] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return np.r_[1.0, vec]

    def predict(self, inputs: Sequence[str]) -> list[float]:
        return [float(self._features(x) @ self.weights) for x in inputs]

    def train(self, clean: Sequence[SourceSample], hard_examples: Sequence[CandidateRecord], *, round_id: int) -> dict:
        rows = [(s.input, s.target) for s in clean if s.target is not None]
        rows += [(r.candidate_input, r.true_target) for r in hard_examples
                 if r.true_target is not None and r.attack_family == "RandomSMILES"]
        x = np.vstack([self._features(text) for text, _ in rows])
        y = np.asarray([target for _, target in rows], dtype=float)
        penalty = np.eye(x.shape[1]) * self.regularization
        penalty[0, 0] = 0.0
        self.weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        pred = x @ self.weights
        self.round_id, self.checkpoint_id = round_id, f"ridge-round-{round_id:03d}"
        self._metrics = {"round_id": round_id, "checkpoint_id": self.checkpoint_id,
                         "training_loss": float(np.mean((pred-y)**2)), "validation_loss": None,
                         "training_data_size": len(rows), "hard_example_count": len(rows)-len(clean), "seed": self.seed,
                         "adapter_warning": "Smoke-only ridge adapter; Transformer fine-tuning remains required for scientific pilot."}
        return dict(self._metrics)

    def evaluate(self, samples: Sequence[SourceSample]) -> dict:
        labelled = [s for s in samples if s.target is not None]
        if not labelled:
            return {"n": 0, "mae": None, "rmse": None, "r2": None}
        y = np.asarray([s.target for s in labelled]); p = np.asarray(self.predict([s.input for s in labelled]))
        residual = y-p; denom = float(np.sum((y-y.mean())**2))
        return {"n": len(y), "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "r2": float(1-np.sum(residual**2)/denom) if denom else None}

    def checkpoint(self, path: str) -> str:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"seed": self.seed, "dimension": self.dimension,
            "regularization": self.regularization, "round_id": self.round_id,
            "checkpoint_id": self.checkpoint_id, "weights": self.weights.tolist(), "metrics": self._metrics}, indent=2))
        return str(target)

    def load_checkpoint(self, path: str) -> None:
        data = json.loads(Path(path).read_text()); self.seed=data["seed"]; self.dimension=data["dimension"]
        self.regularization=data["regularization"]; self.round_id=data["round_id"]
        self.checkpoint_id=data["checkpoint_id"]; self.weights=np.asarray(data["weights"]); self._metrics=data["metrics"]

    def get_metrics(self) -> dict:
        return dict(self._metrics)

