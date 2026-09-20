#!/usr/bin/env python3
"""Recompute matched-k maxima on one fixed source cohort.

The Phase 5C exploratory table admitted every source available at each k. That
changes the population as k increases. This audit holds the >=20-variant source
set fixed and uses nested, without-replacement draws so each Monte Carlo path is
monotone by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUTPUT = ROOT / "results" / "repeat_shift_invariance"
SEED = 20260920
K_VALUES = (1, 2, 5, 10, 20)


def fixed_cohort_matched_k(
    repeat_shift: pd.DataFrame,
    broader: pd.DataFrame,
    *,
    repetitions: int = 5000,
    seed: int = SEED,
    k_values: tuple[int, ...] = K_VALUES,
) -> pd.DataFrame:
    """Estimate expected source maxima on the same sources for every k.

    Each repetition draws one random permutation per source and threat model.
    The k-sample is the first k entries, making samples nested across k. Variant
    sampling is without replacement. Reported standard deviations are across
    Monte Carlo estimates of the cohort-mean source maximum.
    """
    max_k = max(k_values)
    repeat_counts = repeat_shift.groupby("source_id").size()
    broader_counts = broader.groupby("source_id").size()
    eligible = sorted(
        set(repeat_counts[repeat_counts >= max_k].index)
        & set(broader_counts[broader_counts >= max_k].index)
    )
    if not eligible:
        raise ValueError(f"No sources have at least {max_k} variants in both banks")

    rng = np.random.default_rng(seed)
    repeat_estimates = {k: np.empty(repetitions) for k in k_values}
    broader_estimates = {k: np.empty(repetitions) for k in k_values}

    repeat_arrays = {
        source_id: repeat_shift.loc[
            repeat_shift.source_id == source_id, "absolute_drift"
        ].to_numpy()
        for source_id in eligible
    }
    broader_arrays = {
        source_id: broader.loc[
            broader.source_id == source_id, "absolute_drift"
        ].to_numpy()
        for source_id in eligible
    }

    for repetition in range(repetitions):
        repeat_source_maxima = {k: [] for k in k_values}
        broader_source_maxima = {k: [] for k in k_values}
        for source_id in eligible:
            repeat_draw = rng.permutation(repeat_arrays[source_id])[:max_k]
            broader_draw = rng.permutation(broader_arrays[source_id])[:max_k]
            for k in k_values:
                repeat_source_maxima[k].append(float(np.max(repeat_draw[:k])))
                broader_source_maxima[k].append(float(np.max(broader_draw[:k])))
        for k in k_values:
            repeat_estimates[k][repetition] = np.mean(repeat_source_maxima[k])
            broader_estimates[k][repetition] = np.mean(broader_source_maxima[k])

    rows = []
    for k in k_values:
        repeat_mean = float(repeat_estimates[k].mean())
        broader_mean = float(broader_estimates[k].mean())
        rows.append(
            {
                "k": k,
                "n_sources": len(eligible),
                "n_repetitions": repetitions,
                "repeat_shift_mean_expected_max": repeat_mean,
                "repeat_shift_std": float(repeat_estimates[k].std(ddof=1)),
                "broader_mean_expected_max": broader_mean,
                "broader_std": float(broader_estimates[k].std(ddof=1)),
                "paired_difference": broader_mean - repeat_mean,
            }
        )
    result = pd.DataFrame(rows)
    if not result.repeat_shift_mean_expected_max.is_monotonic_increasing:
        raise AssertionError("Fixed-cohort repeat/shift expected maxima are not monotone")
    if not result.broader_mean_expected_max.is_monotonic_increasing:
        raise AssertionError("Fixed-cohort broader expected maxima are not monotone")
    return result


def main() -> None:
    variants = pd.read_csv(OUTPUT / "variants.csv")
    predictions = pd.read_csv(OUTPUT / "predictions.csv")
    repeat_shift = variants[["source_id", "variant_id"]].merge(
        predictions[["source_id", "variant_id", "absolute_drift"]],
        on=["source_id", "variant_id"],
        validate="one_to_one",
    )
    broader = pd.read_csv(
        ROOT / "results" / "representation_equivalence_validation" / "token_pair_audit.csv"
    )[["source_id", "variant_id", "absolute_drift"]]
    result = fixed_cohort_matched_k(repeat_shift, broader)
    result.to_csv(OUTPUT / "matched_k_fixed_cohort.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
