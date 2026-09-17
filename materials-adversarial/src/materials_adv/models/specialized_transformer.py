"""Two-Branch Specialized Transformer model for polymer property regression.

Exposes two explicit learned specialization branches on top of a shared
Transformer encoder:
- Branch A (f_repr): Representation-Invariance Branch
- Branch B (f_chem): Chemistry-Sensitivity Branch

The Bandgap regressor fuses (concatenates) embeddings from both branches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data.scaler import TargetScaler
from ..data.tokenizer import tokenize


class TwoBranchTransformerRegressorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pooling: str = "mean",
        branch_dim: int = 32,
    ) -> None:
        super().__init__()
        # +1 for padding index 0
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.pooling = pooling
        self.branch_dim = branch_dim

        # Branch A: Representation-Invariance Branch (f_repr)
        self.branch_a = nn.Sequential(
            nn.Linear(d_model, branch_dim),
            nn.ReLU(),
            nn.LayerNorm(branch_dim),
        )

        # Branch B: Chemistry-Sensitivity Branch (f_chem)
        self.branch_b = nn.Sequential(
            nn.Linear(d_model, branch_dim),
            nn.ReLU(),
            nn.LayerNorm(branch_dim),
        )

        # Fusion Regressor: Concat(z_repr, z_chem) -> 2 * branch_dim -> 1
        self.regressor = nn.Linear(branch_dim * 2, 1)

    def encode_shared(
        self, src: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the shared pooled sequence representation h before branch splitting."""
        seq_len = src.size(1)
        positions = (
            torch.arange(seq_len, device=src.device)
            .unsqueeze(0)
            .expand(src.size(0), seq_len)
        )

        emb = self.embedding(src) + self.pos_encoder(positions)
        out = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)

        if self.pooling == "mean":
            if padding_mask is not None:
                mask = (~padding_mask).unsqueeze(-1).float()
                out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                out = out.mean(dim=1)
        elif self.pooling == "max":
            if padding_mask is not None:
                out = out.masked_fill(padding_mask.unsqueeze(-1), -float("inf"))
            out = out.max(dim=1)[0]
        elif self.pooling == "cls":
            out = out[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling {self.pooling}")

        return out

    def forward_branches(
        self, src: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tuple of (pred, z_repr, z_chem)."""
        h = self.encode_shared(src, padding_mask=padding_mask)
        z_repr = self.branch_a(h)
        z_chem = self.branch_b(h)
        fusion = torch.cat([z_repr, z_chem], dim=-1)
        pred = self.regressor(fusion).squeeze(-1)
        return pred, z_repr, z_chem

    def forward(
        self, src: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        pred, _, _ = self.forward_branches(src, padding_mask=padding_mask)
        return pred


class TwoBranchTransformerRegressor:
    """Predictor wrapper around TwoBranchTransformerRegressorModel handling tokenization, scaling, and branch extraction."""

    def __init__(
        self,
        model: TwoBranchTransformerRegressorModel,
        vocab: list[str],
        scaler: TargetScaler,
        target_units: str = "eV",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.vocab = vocab
        self.vocab_map = {tok: idx + 1 for idx, tok in enumerate(vocab)}
        self.scaler = scaler
        self.target_units = target_units
        self.device = device
        self.model.eval()

    def _tokenize_batch(
        self, representations: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_sequences = [tokenize(rep) for rep in representations]
        max_len = max(len(seq) for seq in token_sequences) if token_sequences else 1
        max_len = min(max_len, self.model.pos_encoder.num_embeddings)

        batch_size = len(representations)
        tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        mask = torch.ones((batch_size, max_len), dtype=torch.bool)

        for i, seq in enumerate(token_sequences):
            seq_len = min(len(seq), max_len)
            for j in range(seq_len):
                tensor[i, j] = self.vocab_map.get(seq[j], 0)
                mask[i, j] = False

        return tensor.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def predict(self, representations: Sequence[str]) -> list[float]:
        if not representations:
            return []
        src, mask = self._tokenize_batch(representations)
        self.model.eval()
        normalized_preds = self.model(src, padding_mask=mask).cpu().numpy()
        unscaled = self.scaler.inverse_transform(normalized_preds)
        return [float(x) for x in unscaled]

    def predict_with_uncertainty(self, representations: Sequence[str], n_samples: int = 20) -> tuple[list[float], list[float]]:
        """Perform Monte Carlo Dropout (MC-Dropout) to estimate mean prediction and epistemic variance.

        Returns (mean_predictions, epistemic_variances).
        """
        if not representations:
            return [], []
        src, mask = self._tokenize_batch(representations)
        # Enable dropout during inference for MC-Dropout
        self.model.train()
        mc_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                norm_p = self.model(src, padding_mask=mask).cpu().numpy()
                unscaled_p = self.scaler.inverse_transform(norm_p)
                mc_preds.append(unscaled_p)
        self.model.eval()
        
        arr = np.array(mc_preds) # [n_samples, batch_size]
        means = np.mean(arr, axis=0).tolist()
        vars_ = np.var(arr, axis=0).tolist()
        return [float(m) for m in means], [float(v) for v in vars_]


    @torch.no_grad()
    def get_branch_embeddings(
        self, representations: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (predictions_unscaled, z_repr_numpy, z_chem_numpy)."""
        if not representations:
            return (
                np.array([]),
                np.zeros((0, self.model.branch_dim)),
                np.zeros((0, self.model.branch_dim)),
            )
        src, mask = self._tokenize_batch(representations)
        self.model.eval()
        norm_preds, z_repr, z_chem = self.model.forward_branches(
            src, padding_mask=mask
        )
        unscaled = self.scaler.inverse_transform(norm_preds.cpu().numpy())
        return (
            unscaled,
            z_repr.cpu().numpy(),
            z_chem.cpu().numpy(),
        )

    def save(self, directory: str | Path, metrics: dict[str, Any] | None = None) -> Path:
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "model.pt")
        self.scaler.save(out_dir / "scaler.json")
        if metrics is not None:
            with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        return out_dir
