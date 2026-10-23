"""
Dataset utilities for denoising autoencoder training.

Corruption strategies:
  - Random substitution at rate `noise_rate`

Both Bio-Cyber and Materials use the same Dataset class;
the vocab/encode logic is injected.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import pandas as pd


class DenoisingDataset(TorchDataset):
    """
    Each item:
      corrupted_ids   : List[int]  (padded to max_len)
      clean_ids       : List[int]  (padded to max_len)
      tgt_in          : List[int]  (clean[:-1]  — teacher-forced decoder input)
      tgt_out         : List[int]  (clean[1:]   — prediction targets)
    """

    def __init__(
        self,
        sequences: list[list[int]] | str | Path,
        vocab_or_pad: int | dict[str, int],
        vocab_size: int | None = None,
        max_len: int = 256,
        noise_rate: float = 0.15,
        seed: int = 42,
        column: str = "sequence",
        pad_idx: int | None = None,
    ) -> None:
        # Backward-compatible file interface retained for the original tests/tools.
        if isinstance(sequences, (str, Path)):
            vocab = vocab_or_pad
            if not isinstance(vocab, dict):
                raise TypeError("file-backed datasets require a token-to-index vocab")
            frame = pd.read_csv(sequences)
            sequences = [[vocab[c] for c in str(s) if c in vocab] for s in frame[column]]
            pad_idx = 0 if pad_idx is None else pad_idx
            vocab_size = max(vocab.values())
        elif pad_idx is None:
            pad_idx = int(vocab_or_pad)
        if vocab_size is None:
            raise TypeError("vocab_size is required for pre-tokenised sequences")
        self.sequences = [s[:max_len] for s in sequences]
        self.pad_idx = int(pad_idx)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.noise_rate = noise_rate
        self.rng = np.random.default_rng(seed)
        # Build valid token set (1..vocab_size, excluding pad=0)
        self.valid_tokens = [t for t in range(1, vocab_size + 1) if t != pad_idx]

    def _pad(self, ids: list[int]) -> list[int]:
        return ids + [self.pad_idx] * (self.max_len - len(ids))

    def _corrupt(self, ids: list[int]) -> list[int]:
        ids = list(ids)
        for i in range(len(ids)):
            if self.rng.random() < self.noise_rate:
                ids[i] = int(self.rng.choice(self.valid_tokens))
        return ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        clean = self.sequences[idx]
        corrupted = self._corrupt(clean)

        clean_pad = self._pad(clean)
        corr_pad = self._pad(corrupted)

        # teacher forcing: decoder input = [PAD, clean[0],...,clean[-2]]
        tgt_in = [self.pad_idx] + clean_pad[:-1]
        tgt_out = clean_pad  # predict clean tokens at every position

        return (
            torch.tensor(corr_pad, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )
