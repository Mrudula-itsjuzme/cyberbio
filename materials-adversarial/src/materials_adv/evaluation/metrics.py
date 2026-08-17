"""Clean-model regression metrics: MAE, RMSE, R-squared."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination.

    Returns NaN when the target has zero variance -- R^2 is undefined there, and
    returning 0.0 would misrepresent an undefined quantity as a real score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty array")
    return {
        "n": int(y_true.size),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def length_linear_regression(train_lengths: np.ndarray, train_targets: np.ndarray, eval_lengths: np.ndarray, eval_targets: np.ndarray) -> dict[str, float]:
    """Fit a linear regression on sequence length and evaluate it.
    
    This acts as a strict baseline gate. If a complex sequence model cannot
    substantially beat this 1-parameter model, it has merely learned length.
    """
    train_lengths = np.asarray(train_lengths, dtype=float)
    train_targets = np.asarray(train_targets, dtype=float)
    eval_lengths = np.asarray(eval_lengths, dtype=float)
    eval_targets = np.asarray(eval_targets, dtype=float)
    
    # Fit length-only linear regression
    coeffs = np.polyfit(train_lengths, train_targets, 1)
    
    # Predict on eval set
    eval_preds = np.polyval(coeffs, eval_lengths)
    
    metrics = regression_metrics(eval_targets, eval_preds)
    metrics["slope"] = float(coeffs[0])
    metrics["intercept"] = float(coeffs[1])
    return metrics


def length_stratified_metrics(y_true: np.ndarray, y_pred: np.ndarray, lengths: np.ndarray, n_bins: int = 4) -> dict[str, dict[str, float]]:
    """Calculate metrics within length bins to control for length confounds."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    
    if len(y_true) < n_bins:
        # Not enough data to stratify
        return {"all": regression_metrics(y_true, y_pred)}
        
    # Create quantiles
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(lengths, quantiles)
    # Ensure bins are unique (handle many sequences of identical length)
    bins = np.unique(bins)
    
    # If uniqueness collapses bins to < 2 edges, fallback
    if len(bins) < 2:
        return {"all": regression_metrics(y_true, y_pred)}
        
    indices = np.digitize(lengths, bins)
    
    results = {}
    for i in range(1, len(bins)):
        # indices corresponding to the i-th bin
        # digitize: bins[i-1] <= x < bins[i]
        mask = (indices == i)
        # handle inclusivity for the last bin
        if i == len(bins) - 1:
            mask = mask | (lengths == bins[-1])
            
        if np.any(mask):
            bin_name = f"len_{bins[i-1]:.1f}_to_{bins[i]:.1f}"
            results[bin_name] = regression_metrics(y_true[mask], y_pred[mask])
            
    return results
