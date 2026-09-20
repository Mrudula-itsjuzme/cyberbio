import numpy as np
import pandas as pd

from scripts.audit_matched_k_freeze import fixed_cohort_matched_k


def _bank(scales):
    rows = []
    for source_id, scale in scales.items():
        for variant_id, drift in enumerate(np.linspace(0.0, scale, 20)):
            rows.append(
                {
                    "source_id": source_id,
                    "variant_id": variant_id,
                    "absolute_drift": drift,
                }
            )
    return pd.DataFrame(rows)


def test_fixed_cohort_is_constant_and_monotone():
    repeat_shift = _bank({"a": 1.0, "b": 2.0})
    broader = _bank({"a": 2.0, "b": 3.0})
    result = fixed_cohort_matched_k(repeat_shift, broader, repetitions=200, seed=7)
    assert result.n_sources.tolist() == [2] * 5
    assert result.repeat_shift_mean_expected_max.is_monotonic_increasing
    assert result.broader_mean_expected_max.is_monotonic_increasing
    assert (result.paired_difference > 0).all()


def test_sources_below_max_k_are_excluded_at_every_k():
    complete = _bank({"eligible": 1.0})
    incomplete = pd.DataFrame(
        {
            "source_id": ["ineligible"] * 10,
            "variant_id": range(10),
            "absolute_drift": np.linspace(0, 10, 10),
        }
    )
    result = fixed_cohort_matched_k(
        pd.concat([complete, incomplete], ignore_index=True),
        pd.concat([complete, incomplete], ignore_index=True),
        repetitions=50,
        seed=8,
    )
    assert result.n_sources.eq(1).all()


def test_estimator_is_deterministic():
    bank = _bank({"a": 1.0, "b": 2.0})
    first = fixed_cohort_matched_k(bank, bank, repetitions=50, seed=99)
    second = fixed_cohort_matched_k(bank, bank, repetitions=50, seed=99)
    pd.testing.assert_frame_equal(first, second)
