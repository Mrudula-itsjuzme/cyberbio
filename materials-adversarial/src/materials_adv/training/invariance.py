"""Loss and encoding primitives for Phase 6 invariance training."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.tokenizer import tokenize
from ..robustness.equivalence import normalize_attachment_tokens


def anchored_consistency_loss(source_prediction, variant_predictions):
    """Mean squared disagreement from each source prediction."""
    if variant_predictions.ndim == 1:
        variant_predictions = variant_predictions.unsqueeze(1)
    return ((variant_predictions - source_prediction.unsqueeze(1)) ** 2).mean()


def worst_consistency_loss(source_prediction, variant_predictions):
    """Mean across sources of the largest sampled squared disagreement."""
    if variant_predictions.ndim == 1:
        variant_predictions = variant_predictions.unsqueeze(1)
    squared = (variant_predictions - source_prediction.unsqueeze(1)) ** 2
    return squared.max(dim=1).values.mean()


def objective(property_loss, consistency_loss, consistency_weight):
    return property_loss + consistency_weight * consistency_loss


def normalize_and_tokenize(text: str, vocabulary: set[str], max_length: int = 256):
    """Correct the historical bare-star alias and enforce strict eligibility."""
    normalized = normalize_attachment_tokens(text)
    tokens = tokenize(normalized)
    unknown = [token for token in tokens if token not in vocabulary]
    if unknown:
        raise ValueError(f"unknown tokens after attachment normalization: {unknown}")
    if len(tokens) > max_length:
        raise ValueError(f"sequence length {len(tokens)} exceeds {max_length}")
    return normalized, tokens


def encode_batch(texts, vocabulary, max_length=256, device="cpu"):
    mapping = {token: index + 1 for index, token in enumerate(vocabulary)}
    tokenized = [normalize_and_tokenize(text, set(vocabulary), max_length)[1] for text in texts]
    width = max(len(tokens) for tokens in tokenized)
    ids = torch.zeros((len(texts), width), dtype=torch.long, device=device)
    mask = torch.ones((len(texts), width), dtype=torch.bool, device=device)
    for row, tokens in enumerate(tokenized):
        values = [mapping[token] for token in tokens]
        ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, : len(values)] = False
    return ids, mask


@dataclass(frozen=True)
class TrainingAccounting:
    optimizer_steps: int
    source_examples: int
    variant_examples: int
    epochs: int
    wall_seconds: float
