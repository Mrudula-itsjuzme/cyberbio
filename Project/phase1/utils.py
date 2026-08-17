"""
utils.py — Shared helpers for Phase 1: plotting, metrics, serialization.

Part of: Learning to Attack and Defend — Phase 1
"""

import os
import math
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to "TkAgg" locally if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Optional, Tuple

# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for NumPy and Python's random module (+ PyTorch if available)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── Directory helpers ────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    """
    Save a PyTorch model checkpoint.

    Args:
        state: dict with at least keys 'model_state_dict', 'epoch', 'config'.
        path:  file path (e.g. 'checkpoints/baseline.pt').
    """
    try:
        import torch
        ensure_dirs(os.path.dirname(path))
        torch.save(state, path)
        print(f"[Checkpoint] Saved → {path}")
    except ImportError:
        raise RuntimeError("PyTorch is required for save_checkpoint().")


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """
    Load a PyTorch model checkpoint.

    Args:
        path:   file path previously saved by save_checkpoint().
        device: torch device string.

    Returns:
        The checkpoint dict.
    """
    try:
        import torch
        ckpt = torch.load(path, map_location=device)
        print(f"[Checkpoint] Loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
        return ckpt
    except ImportError:
        raise RuntimeError("PyTorch is required for load_checkpoint().")


def save_pickle(obj, path: str) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Pickle] Saved → {path}")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path: str) -> None:
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[JSON] Saved → {path}")


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray,
                       split_name: str = "val") -> Dict[str, float]:
    """Print and return a dict of MAE, RMSE, R² for a given split."""
    mae  = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    r2   = compute_r2(y_true, y_pred)
    print(f"[Metrics | {split_name}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── Plotting ─────────────────────────────────────────────────────────────────

# Shared colour palette
PALETTE = {
    "primary":    "#5C6BC0",   # indigo
    "secondary":  "#26A69A",   # teal
    "accent":     "#EF5350",   # soft red
    "warn":       "#FFA726",   # amber
    "bg":         "#1A1A2E",   # dark navy background
    "grid":       "#2E2E4A",   # subtle grid lines
    "text":       "#E0E0E0",   # light text
}


def _apply_dark_style(fig, axes) -> None:
    """Apply a consistent dark theme to a figure and its axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PALETTE["bg"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        ax.spines["bottom"].set_color(PALETTE["grid"])
        ax.spines["left"].set_color(PALETTE["grid"])
        ax.spines["top"].set_color(PALETTE["grid"])
        ax.spines["right"].set_color(PALETTE["grid"])
        ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.7)


def plot_loss_curve(train_losses: List[float],
                    val_losses: List[float],
                    save_path: str = "outputs/loss_curve.png",
                    title: str = "Baseline Model — Training & Validation Loss") -> str:
    """
    Plot train/val MSE loss over epochs and save to disk.

    Returns:
        Path where the figure was saved.
    """
    ensure_dirs(os.path.dirname(save_path))
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(fig, ax)

    ax.plot(epochs, train_losses, color=PALETTE["primary"],  linewidth=2,
            label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses,   color=PALETTE["secondary"], linewidth=2,
            label="Val Loss",   marker="s", markersize=3, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Loss curve saved → {save_path}")
    return save_path


def plot_drift_distribution(drifts: List[float],
                             save_path: str = "outputs/drift_distribution.png",
                             threshold: float = 0.5,
                             title: str = "Adversarial Prediction Drift Distribution") -> str:
    """
    Histogram of |ŷ_adv − ŷ_orig| drift values, with a threshold line.

    Returns:
        Path where the figure was saved.
    """
    ensure_dirs(os.path.dirname(save_path))
    drifts_arr = np.array(drifts)
    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(fig, ax)

    n_bins = min(40, max(10, len(drifts_arr) // 5))
    ax.hist(drifts_arr, bins=n_bins, color=PALETTE["primary"],
            edgecolor=PALETTE["bg"], alpha=0.85, label="Drift |ŷ_adv − ŷ_orig|")

    ax.axvline(threshold, color=PALETTE["accent"], linewidth=2,
               linestyle="--", label=f"Threshold = {threshold}")
    ax.axvline(float(np.mean(drifts_arr)), color=PALETTE["warn"], linewidth=2,
               linestyle=":", label=f"Mean = {np.mean(drifts_arr):.3f}")

    success_rate = float(np.mean(drifts_arr > threshold)) * 100
    ax.set_xlabel("Prediction Drift", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"{title}\n(Attack success rate @ threshold {threshold}: {success_rate:.1f}%)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Drift distribution saved → {save_path}")
    return save_path


def plot_transition_heatmap(matrix: np.ndarray,
                             tokens: List[str],
                             save_path: str = "outputs/transition_heatmap.png",
                             top_k: int = 20,
                             title: str = "Token Transition Probability Heatmap") -> str:
    """
    Visualise the top-k most frequent tokens' transition sub-matrix as a heatmap.

    Returns:
        Path where the figure was saved.
    """
    ensure_dirs(os.path.dirname(save_path))

    # Select top_k tokens by row-sum (most common source tokens)
    row_sums = matrix.sum(axis=1)
    top_idx  = np.argsort(row_sums)[::-1][:top_k]
    sub      = matrix[np.ix_(top_idx, top_idx)]
    sub_tok  = [tokens[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.55), max(7, top_k * 0.5)))
    _apply_dark_style(fig, ax)

    im = ax.imshow(sub, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    ax.set_xticks(range(len(sub_tok)))
    ax.set_yticks(range(len(sub_tok)))
    ax.set_xticklabels(sub_tok, rotation=45, ha="right", fontsize=9,
                        color=PALETTE["text"])
    ax.set_yticklabels(sub_tok, fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Replacement token (j)", fontsize=11)
    ax.set_ylabel("Source token (i)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Transition heatmap saved → {save_path}")
    return save_path


def plot_token_distribution(token_counts: Dict[str, int],
                             save_path: str = "outputs/token_distribution.png",
                             top_k: int = 30,
                             title: str = "Token Frequency Distribution") -> str:
    """Bar chart of the top-k most frequent tokens in the corpus."""
    ensure_dirs(os.path.dirname(save_path))
    sorted_items = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    toks, cnts   = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_style(fig, ax)

    bar_colors = [PALETTE["primary"]] * len(toks)
    ax.bar(range(len(toks)), cnts, color=bar_colors, edgecolor=PALETTE["bg"], alpha=0.9)
    ax.set_xticks(range(len(toks)))
    ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=9, color=PALETTE["text"])
    ax.set_xlabel("Token", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Token distribution saved → {save_path}")
    return save_path


# ── Results table formatting ─────────────────────────────────────────────────

def format_results_table(records: List[Dict],
                          col_widths: Optional[Dict[str, int]] = None) -> str:
    """
    Pretty-print a list of dicts as a fixed-width table.

    Args:
        records:    list of dicts with consistent keys.
        col_widths: optional {col_name: width} override.

    Returns:
        A multi-line string ready for print().
    """
    if not records:
        return "(no records)"

    keys = list(records[0].keys())
    widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in records))
              for k in keys}
    if col_widths:
        widths.update(col_widths)

    sep  = "─" * (sum(widths.values()) + 3 * len(keys) + 1)
    header = "│ " + " │ ".join(str(k).center(widths[k]) for k in keys) + " │"
    rows   = []
    for r in records:
        row = "│ " + " │ ".join(str(r.get(k, "")).center(widths[k]) for k in keys) + " │"
        rows.append(row)

    return "\n".join([sep, header, sep] + rows + [sep])


# ── Progress bar (no tqdm dependency) ────────────────────────────────────────

def simple_progress(current: int, total: int, prefix: str = "", width: int = 40) -> None:
    """Print an in-place progress bar to stdout."""
    frac  = current / max(total, 1)
    filled = int(frac * width)
    bar   = "█" * filled + "░" * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


# ── __main__ quick-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(42)

    # Test metrics
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yp = np.array([1.1, 1.9, 3.2, 3.8, 5.3])
    regression_report(y, yp, split_name="test")

    # Test table formatter
    records = [
        {"seq": "CCO", "adv_seq": "CNO", "orig_pred": 1.23, "adv_pred": 2.10, "drift": 0.87, "valid": True},
        {"seq": "C=C", "adv_seq": "N=C", "orig_pred": 0.55, "adv_pred": 0.60, "drift": 0.05, "valid": True},
    ]
    print(format_results_table(records))

    # Test loss curve plot
    train_l = [1.0 - 0.03 * i + np.random.rand() * 0.05 for i in range(30)]
    val_l   = [1.1 - 0.025 * i + np.random.rand() * 0.08 for i in range(30)]
    ensure_dirs("outputs")
    plot_loss_curve(train_l, val_l, save_path="outputs/test_loss_curve.png")

    # Test drift distribution plot
    drifts = np.abs(np.random.randn(200)) * 0.6
    plot_drift_distribution(drifts.tolist(), save_path="outputs/test_drift_dist.png")

    print("\n[utils.py] All quick-tests passed.")
