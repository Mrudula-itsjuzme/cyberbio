"""Deterministic, molecule-level evaluation of equivalent SMILES views.

The sampling unit throughout this module is the molecule.  Individual SMILES
views are correlated repeated measurements and must never be treated as
independent observations in confidence intervals or paired comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from .metrics import regression_metrics


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def equivalent_views(smiles: str, n: int, seed: int) -> list[str]:
    """Return exactly ``n`` unique, graph-equivalent randomized views when possible.

    Some small/symmetric molecules have fewer than ``n`` distinct serializations.
    In that case every unique view is returned and the caller records the actual
    count instead of duplicating views and pretending to have more evidence.
    """
    if n < 1:
        raise ValueError("n must be positive")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    views: list[str] = []
    seen: set[str] = set()
    # Multiple deterministic RDKit batches improve coverage without relying on
    # process-global RNG state.  Stop at a fixed limit for reproducibility.
    for offset in range(8):
        batch = Chem.MolToRandomSmilesVect(
            mol, max(n * 4, 16), randomSeed=int(seed + offset)
        )
        for view in batch:
            parsed = Chem.MolFromSmiles(view)
            if parsed is None:
                continue
            if Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True) != identity:
                raise AssertionError("RDKit randomized view changed canonical identity")
            if view not in seen:
                seen.add(view)
                views.append(view)
                if len(views) == n:
                    return views
    return views


def trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if not 0 <= proportion < 0.5:
        raise ValueError("trim proportion must be in [0, 0.5)")
    k = int(np.floor(len(values) * proportion))
    if k == 0:
        return float(np.mean(values))
    return float(np.mean(values[k:-k]))


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(xr, yr)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def molecule_descriptors(smiles: str, views: Iterable[str]) -> dict[str, float | str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    atoms = list(mol.GetAtoms())
    heavy = [a for a in atoms if a.GetAtomicNum() > 1]
    aromatic = sum(int(a.GetIsAromatic()) for a in heavy)
    view_list = list(views)
    lengths = np.asarray([len(v) for v in view_list], dtype=float)
    tokens = set("".join(view_list))
    scaffold = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return {
        "smiles_length": float(len(smiles)),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "ring_count": float(Lipinski.RingCount(mol)),
        "aromatic_fraction": float(aromatic / len(heavy)) if heavy else 0.0,
        "heteroatom_count": float(sum(a.GetAtomicNum() not in (0, 1, 6) for a in atoms)),
        "attachment_point_count": float(sum(a.GetAtomicNum() == 0 for a in atoms)),
        "token_diversity": float(len(tokens)),
        "representation_length_variance": float(np.var(lengths)) if len(lengths) else 0.0,
        "scaffold_family": scaffold,
    }


def evaluate_multiview(
    frame: pd.DataFrame,
    predict: Callable[[list[str]], np.ndarray],
    *,
    n_views: int = 16,
    seed: int = 20260921,
) -> pd.DataFrame:
    """Evaluate canonical and equivalent-view aggregate predictions per molecule."""
    records: list[dict] = []
    for row_number, row in frame.reset_index(drop=True).iterrows():
        smiles = str(row["original_representation"])
        target = float(row["property_value"])
        canonical = canonical_smiles(smiles)
        views = equivalent_views(smiles, n_views, seed + row_number * 1009)
        if not views:
            views = [canonical]
        source_pred = float(np.asarray(predict([smiles])).reshape(-1)[0])
        canonical_pred = float(np.asarray(predict([canonical])).reshape(-1)[0])
        predictions = np.asarray(predict(views), dtype=float).reshape(-1)
        if len(predictions) != len(views) or not np.all(np.isfinite(predictions)):
            raise ValueError("predictor returned invalid multi-view predictions")
        mean_pred = float(np.mean(predictions))
        median_pred = float(np.median(predictions))
        trim_pred = trimmed_mean(predictions)
        record = {
            "polymer_id": row.get("polymer_id", str(row_number)),
            "original_representation": smiles,
            "canonical_representation": canonical,
            "target_eV": target,
            "requested_views": n_views,
            "unique_views": len(views),
            "source_prediction_eV": source_pred,
            "canonical_prediction_eV": canonical_pred,
            "mean_prediction_eV": mean_pred,
            "median_prediction_eV": median_pred,
            "trimmed_mean_prediction_eV": trim_pred,
            "source_abs_error_eV": abs(source_pred - target),
            "canonical_abs_error_eV": abs(canonical_pred - target),
            "mean_abs_error_eV": abs(mean_pred - target),
            "prediction_std_eV": float(np.std(predictions, ddof=0)),
            "prediction_range_eV": float(np.ptp(predictions)),
            "worst_case_canonical_drift_eV": float(np.max(np.abs(predictions - canonical_pred))),
            "mean_canonical_drift_eV": float(np.mean(np.abs(predictions - canonical_pred))),
        }
        record.update(molecule_descriptors(smiles, views))
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_multiview(per_molecule: pd.DataFrame) -> dict:
    y = per_molecule["target_eV"].to_numpy(float)
    summary: dict[str, object] = {"n_molecules": int(len(per_molecule))}
    for name, col in {
        "dataset_source": "source_prediction_eV",
        "canonical": "canonical_prediction_eV",
        "mean": "mean_prediction_eV",
        "median": "median_prediction_eV",
        "trimmed_mean": "trimmed_mean_prediction_eV",
    }.items():
        summary[name] = regression_metrics(y, per_molecule[col].to_numpy(float))
    error = per_molecule["mean_abs_error_eV"].to_numpy(float)
    correlations = {}
    for signal in ("prediction_std_eV", "prediction_range_eV", "worst_case_canonical_drift_eV"):
        x = per_molecule[signal].to_numpy(float)
        correlations[signal] = {
            "pearson": _pearson(x, error),
            "spearman": _rank_correlation(x, error),
        }
    summary["disagreement_error_correlations"] = correlations
    summary["representation_drift"] = {
        "molecule_mean_eV": float(per_molecule["mean_canonical_drift_eV"].mean()),
        "molecule_median_eV": float(per_molecule["mean_canonical_drift_eV"].median()),
        "molecule_p90_eV": float(per_molecule["mean_canonical_drift_eV"].quantile(0.9)),
        "worst_observed_eV": float(per_molecule["worst_case_canonical_drift_eV"].max()),
        "prediction_variance_mean_eV2": float(np.mean(per_molecule["prediction_std_eV"] ** 2)),
    }
    disagreement = per_molecule["prediction_std_eV"]
    low = per_molecule[disagreement <= disagreement.quantile(0.25)]
    high = per_molecule[disagreement >= disagreement.quantile(0.75)]
    summary["disagreement_group_comparison"] = {
        "definition": "bottom versus top quartile of per-molecule equivalent-view prediction SD",
        "error_threshold_eV": 0.5,
        "low_n": int(len(low)),
        "high_n": int(len(high)),
        "low_mean_abs_error_eV": float(low["mean_abs_error_eV"].mean()),
        "high_mean_abs_error_eV": float(high["mean_abs_error_eV"].mean()),
        "low_error_rate": float(np.mean(low["mean_abs_error_eV"] > 0.5)),
        "high_error_rate": float(np.mean(high["mean_abs_error_eV"] > 0.5)),
    }
    descriptor_correlations = {}
    for descriptor in (
        "smiles_length", "molecular_weight", "ring_count", "aromatic_fraction",
        "heteroatom_count", "attachment_point_count", "token_diversity",
        "representation_length_variance",
    ):
        x = per_molecule[descriptor].to_numpy(float)
        drift = per_molecule["worst_case_canonical_drift_eV"].to_numpy(float)
        descriptor_correlations[descriptor] = {
            "pearson": _pearson(x, drift),
            "spearman": _rank_correlation(x, drift),
        }
    summary["failure_mode_correlations"] = descriptor_correlations
    return summary


def molecule_bootstrap_mae_delta(
    per_molecule: pd.DataFrame,
    *,
    aggregate_col: str = "mean_prediction_eV",
    seed: int = 20260921,
    n_resamples: int = 2000,
) -> dict[str, float | int | str]:
    """Paired molecule bootstrap of aggregate MAE minus canonical MAE."""
    y = per_molecule["target_eV"].to_numpy(float)
    canonical = per_molecule["canonical_prediction_eV"].to_numpy(float)
    aggregate = per_molecule[aggregate_col].to_numpy(float)
    per_molecule_delta = np.abs(aggregate - y) - np.abs(canonical - y)
    rng = np.random.default_rng(seed)
    n = len(per_molecule_delta)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        draws[i] = float(np.mean(per_molecule_delta[rng.integers(0, n, size=n)]))
    return {
        "unit": "molecule",
        "n_molecules": n,
        "n_resamples": n_resamples,
        "seed": seed,
        "mean_mae_delta_eV": float(np.mean(per_molecule_delta)),
        "ci95_low_eV": float(np.quantile(draws, 0.025)),
        "ci95_high_eV": float(np.quantile(draws, 0.975)),
    }


def disagreement_bins(per_molecule: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Calibration-style bins; rows remain molecules, never individual views."""
    out = per_molecule[["prediction_std_eV", "mean_abs_error_eV"]].copy()
    out["bin"] = pd.qcut(out["prediction_std_eV"], q=n_bins, duplicates="drop")
    grouped = out.groupby("bin", observed=True)
    return grouped.agg(
        n_molecules=("mean_abs_error_eV", "size"),
        disagreement_mean_eV=("prediction_std_eV", "mean"),
        error_mean_eV=("mean_abs_error_eV", "mean"),
        error_median_eV=("mean_abs_error_eV", "median"),
    ).reset_index().assign(bin=lambda x: x["bin"].astype(str))
