from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate
import yaml

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.training.train import PolymerDataset, trim_padded_batch

from .contracts import CandidateRecord, Defender, SourceSample


class RepresentationPairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], vocab: list[str], max_len: int):
        self.pairs = pairs
        self.char2idx = {token: index + 1 for index, token in enumerate(vocab)}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, representation: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.char2idx.get(token, 0) for token in tokenize(representation)][: self.max_len]
        mask = [False] * len(ids)
        ids.extend([0] * (self.max_len - len(ids)))
        mask.extend([True] * (self.max_len - len(mask)))
        return torch.tensor(ids), torch.tensor(mask, dtype=torch.bool)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        clean, adversarial = self.pairs[index]
        clean_ids, clean_mask = self._encode(clean)
        adv_ids, adv_mask = self._encode(adversarial)
        return {"clean_ids": clean_ids, "clean_mask": clean_mask,
                "adv_ids": adv_ids, "adv_mask": adv_mask}


def _trim_pair_batch(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = default_collate(rows)
    for prefix in ("clean", "adv"):
        width = max(1, int((~batch[f"{prefix}_mask"]).sum(dim=1).max().item()))
        batch[f"{prefix}_ids"] = batch[f"{prefix}_ids"][:, :width]
        batch[f"{prefix}_mask"] = batch[f"{prefix}_mask"][:, :width]
    return batch


class TransPolymerDefender(Defender):
    """Warm-started adapter around the canonical Transformer regressor.

    Chemistry-changing hard examples are diagnostics only. Representation-
    preserving RandomSMILES examples contribute prediction-consistency loss;
    their source labels are also valid and may contribute supervised loss.
    """

    def __init__(self, project_root: str | Path, *, seed: int, checkpoint_dir: str | Path,
                 epochs: int = 1, learning_rate: float | None = None,
                 batch_size: int | None = None, lambda_rep: float = 0.1,
                 device: str = "cpu"):
        self.project_root = Path(project_root)
        self.seed, self.epochs, self.lambda_rep = seed, epochs, lambda_rep
        self.device = torch.device(device)
        self.config = yaml.safe_load((self.project_root / "configs/model.yaml").read_text())
        self.data_config = yaml.safe_load((self.project_root / "configs/dataset.yaml").read_text())
        self.vocab = json.loads((self.project_root / "data/processed/vocab.json").read_text())
        self.splits = json.loads((self.project_root / "data/processed/splits.json").read_text())
        self.scaler = TargetScaler.load(self.project_root / checkpoint_dir / "scaler.json")
        self.learning_rate = learning_rate or float(self.config["training"]["learning_rate"])
        self.batch_size = batch_size or int(self.config["training"]["batch_size"])
        self.model = self._new_model()
        state = torch.load(self.project_root / checkpoint_dir / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.predictor = TransformerRegressor(self.model, self.vocab, self.scaler,
                                               self.data_config["target_units"])
        self.round_id = 0
        self.checkpoint_id = "canonical-D0"
        self._metrics: dict = {"seed": seed, "checkpoint_id": self.checkpoint_id}
        self._initial_checkpoint = str(checkpoint_dir)

    def _new_model(self) -> TransformerRegressorModel:
        arch = self.config["architecture"]
        return TransformerRegressorModel(
            vocab_size=len(self.vocab), d_model=arch["d_model"], n_layers=arch["n_layers"],
            n_heads=arch["n_heads"], dim_feedforward=arch["dim_feedforward"],
            dropout=arch["dropout"], max_seq_len=arch["max_seq_len"], pooling=arch["pooling"],
        ).to(self.device)

    def predict(self, inputs: Sequence[str]) -> list[float]:
        return self.predictor.predict(list(inputs)).astype(float).tolist()

    def evaluate(self, samples: Sequence[SourceSample]) -> dict[str, float | int | None]:
        labelled = [sample for sample in samples if sample.target is not None]
        if not labelled:
            return {"n": 0, "mae": None, "rmse": None, "r2": None}
        target = np.asarray([sample.target for sample in labelled], dtype=float)
        prediction = np.asarray(self.predict([sample.input for sample in labelled]))
        residual = target - prediction
        denominator = float(np.sum((target - target.mean()) ** 2))
        return {"n": len(labelled), "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
                "r2": float(1 - np.sum(residual ** 2) / denominator) if denominator else None}

    def train(self, clean: Sequence[SourceSample], hard_examples: Sequence[CandidateRecord], *, round_id: int) -> dict:
        random.seed(self.seed + round_id); np.random.seed(self.seed + round_id); torch.manual_seed(self.seed + round_id)
        clean_frame = pd.DataFrame({"psmiles": [s.input for s in clean if s.target is not None],
                                    "target": [s.target for s in clean if s.target is not None]})
        clean_dataset = PolymerDataset(clean_frame, self.vocab,
                                       self.config["architecture"]["max_seq_len"], self.scaler)
        clean_loader = DataLoader(clean_dataset, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=trim_padded_batch)
        representation_records = [record for record in hard_examples
                                  if record.attack_family == "RandomSMILES"
                                  and record.validity_status == "valid"]
        pairs = [(record.original_input, record.candidate_input) for record in representation_records]
        pair_loader = None
        if pairs:
            pair_loader = DataLoader(RepresentationPairDataset(pairs, self.vocab,
                self.config["architecture"]["max_seq_len"]), batch_size=min(self.batch_size, len(pairs)),
                shuffle=True, collate_fn=_trim_pair_batch)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                      weight_decay=float(self.config["training"]["weight_decay"]))
        history, best_state, best_val = [], copy.deepcopy(self.model.state_dict()), float("inf")
        validation = self._validation_samples()
        for epoch in range(self.epochs):
            self.model.train(); total = clean_total = rep_total = 0.0; batches = 0
            pair_iterator = iter(pair_loader) if pair_loader else None
            for batch in clean_loader:
                optimizer.zero_grad()
                ids, mask, target = batch["ids"].to(self.device), batch["mask"].to(self.device), batch["target"].to(self.device)
                clean_loss = F.mse_loss(self.model(ids, padding_mask=mask), target)
                rep_loss = torch.tensor(0.0, device=self.device)
                if pair_iterator is not None:
                    try: pair = next(pair_iterator)
                    except StopIteration:
                        pair_iterator = iter(pair_loader); pair = next(pair_iterator)
                    clean_prediction = self.model(pair["clean_ids"].to(self.device), padding_mask=pair["clean_mask"].to(self.device))
                    adversarial_prediction = self.model(pair["adv_ids"].to(self.device), padding_mask=pair["adv_mask"].to(self.device))
                    rep_loss = F.mse_loss(adversarial_prediction, clean_prediction.detach())
                loss = clean_loss + self.lambda_rep * rep_loss
                loss.backward(); optimizer.step()
                total += float(loss.detach()); clean_total += float(clean_loss.detach()); rep_total += float(rep_loss.detach()); batches += 1
            val_metrics = self.evaluate(validation)
            history.append({"epoch": epoch + 1, "training_loss": total / batches,
                            "clean_loss": clean_total / batches, "representation_loss": rep_total / batches,
                            "validation_mae": val_metrics["mae"]})
            if val_metrics["mae"] < best_val:
                best_val = float(val_metrics["mae"]); best_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state); self.model.eval()
        self.round_id = round_id; self.checkpoint_id = f"transformer-D{round_id:03d}"
        final_validation = self.evaluate(validation)
        self._metrics = {"round_id": round_id, "checkpoint_id": self.checkpoint_id, "seed": self.seed,
            "train_split": "canonical train", "validation_split": "canonical validation",
            "clean_mae": final_validation["mae"], "clean_rmse": final_validation["rmse"],
            "clean_r2": final_validation["r2"], "training_loss": history[-1]["training_loss"],
            "validation_loss": history[-1]["validation_mae"], "training_data_size": len(clean_dataset),
            "hard_example_count": len(hard_examples), "representation_hard_example_count": len(pairs),
            "diagnostic_only_chemistry_changing_count": len(hard_examples) - len(pairs),
            "epochs": self.epochs, "learning_rate": self.learning_rate, "batch_size": self.batch_size,
            "optimizer": "AdamW", "loss_terms": {"clean_supervised_mse": 1.0,
                "representation_consistency_mse": self.lambda_rep,
                "chemistry_changing_supervised": 0.0}, "history": history}
        return dict(self._metrics)

    def _validation_samples(self) -> list[SourceSample]:
        frame = pd.read_csv(self.project_root / "data/processed/processed.csv")
        rows = frame.iloc[self.splits["val"]]
        return [SourceSample(str(row.polymer_id), row.original_representation, float(row.property_value))
                for row in rows.itertuples()]

    def checkpoint(self, path: str) -> str:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "metrics": self._metrics,
                    "seed": self.seed, "round_id": self.round_id,
                    "checkpoint_id": self.checkpoint_id, "config": self.config,
                    "scaler": {"mean": self.scaler.mean, "std": self.scaler.std}}, target)
        return str(target)

    def load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"]); self.model.eval()
        self.round_id = int(payload["round_id"]); self.checkpoint_id = payload["checkpoint_id"]
        self._metrics = payload["metrics"]

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def provenance(self) -> dict:
        checkpoint = self.project_root / self._initial_checkpoint / "model.pt"
        return {"initial_checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "architecture": self.config["architecture"], "scaler": {"mean": self.scaler.mean, "std": self.scaler.std}}
