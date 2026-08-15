"""Record schema, persistence, and metric tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from materials_adv.evaluation.attack_metrics import (
    SuccessCriterion,
    group_by_attack_type,
    summarize_attacks,
)
from materials_adv.evaluation.metrics import mae, r2, regression_metrics, rmse
from materials_adv.evaluation.records import (
    SCHEMA_VERSION,
    AttackRecord,
    ValidityStatus,
    compute_drift,
    make_attack_id,
)
from materials_adv.utils.io import read_jsonl, write_jsonl

REQUIRED_FIELDS = {
    "attack_id",
    "sample_id",
    "original_psmiles",
    "adversarial_psmiles",
    "attack_type",
    "number_of_changes",
    "validity_status",
    "original_prediction",
    "adversarial_prediction",
    "prediction_drift",
}

FUTURE_FIELDS = {
    "attack_score",
    "confidence_change",
    "uncertainty_change",
    "chemical_constraint_score",
    "sampling_probability",
}


def make_record(**overrides) -> AttackRecord:
    base = {
        "attack_id": "s1:deletion:0:0",
        "sample_id": "s1",
        "original_psmiles": "[*]CC[*]",
        "adversarial_psmiles": "[*]C[*]",
        "attack_type": "deletion",
        "number_of_changes": 1,
        "validity_status": ValidityStatus.VALID.value,
    }
    return AttackRecord(**{**base, **overrides})


# --- Schema ------------------------------------------------------------------


def test_record_has_all_required_fields() -> None:
    assert REQUIRED_FIELDS <= set(make_record().to_dict())


def test_future_fields_exist_now_as_null() -> None:
    """Present from the start so the on-disk schema never migrates mid-project."""
    d = make_record().to_dict()
    assert FUTURE_FIELDS <= set(d)
    assert all(d[f] is None for f in FUTURE_FIELDS)


def test_record_is_frozen() -> None:
    with pytest.raises((AttributeError, TypeError)):
        make_record().attack_id = "mutated"  # type: ignore[misc]


def test_schema_version_recorded() -> None:
    assert make_record().schema_version == SCHEMA_VERSION


def test_attack_id_is_deterministic() -> None:
    a = make_attack_id("s1", "deletion", 42, 0)
    assert a == make_attack_id("s1", "deletion", 42, 0)
    assert a != make_attack_id("s1", "deletion", 43, 0)


def test_attack_id_handles_missing_seed() -> None:
    assert "noseed" in make_attack_id("s1", "deletion", None, 0)


# --- Signed drift ------------------------------------------------------------


def test_drift_is_signed_not_absolute() -> None:
    """Direction is a real finding; abs() belongs at reporting time only."""
    assert compute_drift(300.0, 250.0) == -50.0
    assert compute_drift(300.0, 350.0) == 50.0


def test_drift_none_when_prediction_missing() -> None:
    assert compute_drift(None, 250.0) is None
    assert compute_drift(300.0, None) is None


# --- Persistence -------------------------------------------------------------


def test_jsonl_round_trip(tmp_path) -> None:
    records = [make_record(attack_id=f"s1:deletion:0:{i}") for i in range(3)]
    path = tmp_path / "attacks.jsonl"
    assert write_jsonl(path, (r.to_dict() for r in records)) == 3
    restored = [AttackRecord.from_dict(d) for d in read_jsonl(path)]
    assert [r.attack_id for r in restored] == [r.attack_id for r in records]
    assert restored[0] == records[0]


def test_read_jsonl_reports_malformed_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSONL"):
        list(read_jsonl(path))


# --- Regression metrics ------------------------------------------------------


def test_regression_metrics_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["r2"] == 1.0


def test_mae_and_rmse_differ_with_outlier() -> None:
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0, 4.0])
    assert mae(y_true, y_pred) == 1.0
    assert rmse(y_true, y_pred) == 2.0


def test_r2_is_nan_for_zero_variance_target() -> None:
    """R^2 is undefined here; returning 0.0 would misreport it as a real score."""
    assert math.isnan(r2(np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 6.0])))


def test_metrics_reject_empty_and_mismatched() -> None:
    with pytest.raises(ValueError, match="empty"):
        regression_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="shape mismatch"):
        regression_metrics(np.array([1.0]), np.array([1.0, 2.0]))


# --- Attack metrics ----------------------------------------------------------


def test_success_is_not_merely_prediction_changed() -> None:
    """A tiny drift below threshold must not count as a success."""
    record = make_record(original_prediction=300.0, adversarial_prediction=300.5,
                         prediction_drift=0.5)
    assert SuccessCriterion(min_abs_drift=10.0).is_success(record) is False
    assert SuccessCriterion(min_abs_drift=0.1).is_success(record) is True


def test_unchecked_is_not_success_under_strict_policy() -> None:
    record = make_record(
        validity_status=ValidityStatus.UNCHECKED.value,
        prediction_drift=50.0,
    )
    assert SuccessCriterion(min_abs_drift=1.0).is_success(record) is False
    assert (
        SuccessCriterion(min_abs_drift=1.0, count_unchecked_as_valid=True).is_success(record)
        is True
    )


def test_perturbation_size_constrains_success() -> None:
    record = make_record(number_of_changes=5, prediction_drift=50.0)
    assert SuccessCriterion(min_abs_drift=1.0, max_changes=1).is_success(record) is False


def test_summary_reports_failures_alongside_successes() -> None:
    records = [
        make_record(attack_id="a", prediction_drift=50.0),
        make_record(attack_id="b", validity_status=ValidityStatus.INVALID_REPRESENTATION.value),
        make_record(attack_id="c", validity_status=ValidityStatus.UNCHECKED.value),
    ]
    s = summarize_attacks(records, SuccessCriterion(min_abs_drift=1.0))
    assert s["n_total"] == 3
    assert s["n_invalid"] == 1 and s["n_unchecked"] == 1
    assert s["n_unscored"] == 2  # only one record carries a drift value


def test_summary_reports_signed_mean_to_expose_directional_bias() -> None:
    records = [
        make_record(attack_id="a", prediction_drift=-50.0),
        make_record(attack_id="b", prediction_drift=-30.0),
    ]
    s = summarize_attacks(records)
    assert s["mean_signed_drift"] == -40.0
    assert s["mean_abs_drift"] == 40.0


def test_empty_summary_is_safe() -> None:
    assert summarize_attacks([])["n_total"] == 0


def test_group_by_attack_type() -> None:
    records = [
        make_record(attack_id="a", attack_type="deletion"),
        make_record(attack_id="b", attack_type="insertion"),
    ]
    grouped = group_by_attack_type(records)
    assert set(grouped) == {"deletion", "insertion"}
