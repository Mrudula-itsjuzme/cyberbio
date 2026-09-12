"""Metrics and lightweight controls for representation-shortcut attribution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from ..data.tokenizer import tokenize


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays with equal shape")
    residual = y - p
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


@dataclass
class RidgeRegressor:
    alpha: float = 1.0
    coefficients_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: Sequence[float]) -> "RidgeRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([x, np.ones(len(x))])
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        self.coefficients_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(x, dtype=float)
        return np.column_stack([x, np.ones(len(x))]) @ self.coefficients_


def length_features(strings: Sequence[str]) -> np.ndarray:
    return np.asarray([[len(tokenize(s))] for s in strings], dtype=float)


def token_features(strings: Sequence[str], vocab: Sequence[str], *, frequencies: bool) -> np.ndarray:
    index = {token: i for i, token in enumerate(vocab)}
    matrix = np.zeros((len(strings), len(vocab)), dtype=float)
    for row, text in enumerate(strings):
        counts = Counter(tokenize(text))
        total = sum(counts.values())
        for token, count in counts.items():
            if token in index:
                matrix[row, index[token]] = count / total if frequencies and total else count
    return matrix


def safe_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def representation_invariance_score(predictions: Sequence[float], reference_scale: float) -> float:
    """Unitless 1/(1 + prediction SD/reference scale); 1 means invariant."""
    values = np.asarray(predictions, dtype=float)
    if values.size == 0 or reference_scale <= 0:
        raise ValueError("predictions must be non-empty and reference_scale positive")
    return float(1.0 / (1.0 + np.std(values, ddof=0) / reference_scale))


def representation_statistics(
    predictions: Sequence[float], lengths: Sequence[int], embeddings: np.ndarray,
    *, reference_scale: float,
) -> dict[str, float | int | None]:
    p = np.asarray(predictions, dtype=float)
    e = np.asarray(embeddings, dtype=float)
    if p.size != len(lengths) or e.shape[0] != p.size or p.size == 0:
        raise ValueError("predictions, lengths, and embedding rows must align")
    pairwise = [float(np.linalg.norm(e[i] - e[j])) for i, j in combinations(range(len(e)), 2)]
    return {
        "n_representations": int(p.size), "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()), "prediction_range": float(np.ptp(p)),
        "max_pairwise_prediction_drift": float(np.ptp(p)),
        "length_prediction_correlation": safe_correlation(lengths, p),
        "embedding_mean_feature_variance": float(np.var(e, axis=0).mean()),
        "embedding_mean_pairwise_distance": float(np.mean(pairwise)) if pairwise else 0.0,
        "embedding_max_pairwise_distance": float(np.max(pairwise)) if pairwise else 0.0,
        "representation_invariance_score": representation_invariance_score(p, reference_scale),
    }


def paired_bootstrap_ci(
    values: Sequence[float], *, rng: np.random.Generator, confidence: float = 0.95,
    n_resamples: int = 2000, minimum_n: int = 20,
) -> dict[str, float | int | str | None]:
    data = np.asarray(values, dtype=float)
    if len(data) < minimum_n:
        return {"n": int(len(data)), "mean": float(data.mean()) if len(data) else None,
                "ci_low": None, "ci_high": None, "status": "insufficient_sample"}
    samples = rng.choice(data, size=(n_resamples, len(data)), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {"n": int(len(data)), "mean": float(data.mean()),
            "ci_low": float(np.quantile(samples, tail)),
            "ci_high": float(np.quantile(samples, 1.0 - tail)), "status": "estimated"}


def occlusion_sensitivity(tokens: Sequence[str], predict: Callable[[list[list[str]]], np.ndarray]) -> list[dict]:
    """Leave one token out. Outputs are diagnostic and need not be valid SMILES."""
    base = float(predict([list(tokens)])[0])
    variants = [list(tokens[:i]) + list(tokens[i + 1 :]) for i in range(len(tokens))]
    if not variants:
        return []
    changed = predict(variants)
    return [{"position": i, "token": token, "prediction": float(value),
             "signed_delta": float(value - base), "absolute_delta": float(abs(value - base))}
            for i, (token, value) in enumerate(zip(tokens, changed))]
