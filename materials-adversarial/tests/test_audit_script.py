"""Audit-script tests.

SYNTHETIC FIXTURES ONLY. Every table here is built inside the test with planted
defects, to verify the DETECTOR works. None of it is polymer data, none of it is
written to data/, and none of it may ever be used for training or reported as a
result.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_dataset.py"

spec = importlib.util.spec_from_file_location("audit_dataset", SCRIPT)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


# --- The blocker must be detected, not worked around -------------------------


def test_empty_directory_exits_nonzero_with_guidance(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "BLOCKED" in result.stderr
    assert "Openpoly_benchmark" in result.stderr
    assert "No data will be fabricated" in result.stderr


def test_real_data_dir_is_currently_empty() -> None:
    """Documents the live blocker: if this fails, the data has arrived."""
    assert audit.discover_files(ROOT / "data" / "raw") == []


# --- Representation-column discovery -----------------------------------------


def test_psmiles_column_scores_far_above_other_text() -> None:
    """Scoring uses the tokenizer, not column names."""
    df = pd.DataFrame(
        {
            "structure": ["[*]CC(=O)O[*]", "[*]CCO[*]", "*CC(c1ccccc1)*"],
            "notes": ["measured by DSC", "literature value", "estimated"],
        }
    )
    rep = audit.score_representation_column(df["structure"])
    other = audit.score_representation_column(df["notes"])
    assert rep["score"] > 0.8
    assert rep["frac_containing_attachment_star"] == 1.0
    assert rep["score"] > other["score"]


def test_representation_scoring_ignores_column_name() -> None:
    """A PSMILES column named something unhelpful must still be found."""
    series = pd.Series(["[*]CC[*]", "[*]CCO[*]", "[*]CCN[*]"])
    assert audit.score_representation_column(series)["score"] > 0.8


def test_empty_column_scores_zero() -> None:
    assert audit.score_representation_column(pd.Series([None, ""], dtype=object))["score"] == 0.0


def test_text_columns_detected_across_pandas_versions() -> None:
    """Regression: pandas 3.0 uses a 'str' dtype, not 'object'.

    A `dtype == object` check finds nothing on pandas 3.x, so the audit would
    silently report "no representation column" for a perfectly good file.
    """
    df = pd.DataFrame({"psmiles": ["[*]CC[*]"], "tg": [350.0], "flag": [True]})
    assert audit._is_text_dtype(df["psmiles"]) is True
    assert audit._is_text_dtype(df["tg"]) is False
    assert audit._is_text_dtype(df["flag"]) is False
    # Explicit object dtype must still work (pandas 2.x path).
    assert audit._is_text_dtype(pd.Series(["[*]CC[*]"], dtype=object)) is True


# --- Tg unit detection -------------------------------------------------------


def test_negative_values_rule_out_kelvin() -> None:
    result = audit.analyze_tg_units(np.array([-50.0, -20.0, 10.0, 80.0, 120.0]))
    assert result["verdict"] == "likely_celsius"
    assert "rule out pure Kelvin" in result["reason"]
    assert result["requires_human_confirmation"] is True


def test_all_high_positive_reads_as_kelvin() -> None:
    rng = np.random.default_rng(0)
    result = audit.analyze_tg_units(rng.normal(400, 40, 300).clip(200, 600))
    assert result["verdict"] == "likely_kelvin"
    assert result["requires_human_confirmation"] is True


def test_planted_kelvin_celsius_mixture_is_flagged() -> None:
    """The defect the naive largest-gap probe failed to catch."""
    rng = np.random.default_rng(1)
    celsius = rng.normal(100, 30, 200)
    kelvin = celsius + 273.15
    result = audit.analyze_tg_units(np.concatenate([celsius, kelvin]))
    assert result["verdict"] in {"possibly_mixed", "likely_celsius"}
    assert result["requires_human_confirmation"] is True


def test_unit_analysis_never_converts_values() -> None:
    """Evidence only: the reported min/max must be the raw input."""
    values = np.array([-10.0, 50.0, 120.0])
    result = audit.analyze_tg_units(values)
    assert result["min"] == -10.0 and result["max"] == 120.0


def test_empty_values_are_undetermined() -> None:
    assert audit.analyze_tg_units(np.array([]))["verdict"] == "undetermined"


# --- Duplicate and conflict detection ----------------------------------------


def test_planted_conflicting_duplicate_is_reported_not_resolved() -> None:
    df = pd.DataFrame(
        {
            "psmiles": ["[*]CC[*]", "[*]CC[*]", "[*]CCO[*]"],
            "Tg": [350.0, 380.0, 400.0],
        }
    )
    result = audit.find_duplicate_conflicts(df, "psmiles", "Tg")
    assert result["n_conflicting_representations"] == 1
    assert result["max_conflict_spread"] == 30.0
    assert "NOT AUTO-RESOLVED" in result["resolution_policy"]


def test_conflict_differing_by_273_is_flagged_as_unit_mix() -> None:
    """The most diagnostic conflict: same polymer recorded in K and in C."""
    df = pd.DataFrame(
        {"psmiles": ["[*]CC[*]", "[*]CC[*]"], "Tg": [100.0, 373.15]}
    )
    result = audit.find_duplicate_conflicts(df, "psmiles", "Tg")
    assert result["conflict_examples"][0]["differs_by_approx_273.15"] is True


def test_exact_duplicates_are_not_counted_as_conflicts() -> None:
    df = pd.DataFrame({"psmiles": ["[*]CC[*]", "[*]CC[*]"], "Tg": [350.0, 350.0]})
    result = audit.find_duplicate_conflicts(df, "psmiles", "Tg")
    assert result["n_representations_appearing_more_than_once"] == 1
    assert result["n_conflicting_representations"] == 0


def test_near_duplicates_detected_after_normalization() -> None:
    df = pd.DataFrame({"psmiles": ["[*]CC[*]", " [*]CC[*] "], "Tg": [350.0, 360.0]})
    result = audit.find_duplicate_conflicts(df, "psmiles", "Tg")
    assert result["n_near_duplicates_after_normalization"] >= 1


# --- Full-file audit ---------------------------------------------------------


def test_audit_proposes_columns_but_never_confirms(tmp_path) -> None:
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "psmiles": ["[*]CC(=O)O[*]", "[*]CCO[*]", "[*]CCN[*]"],
            "Tg_K": [350.0, 360.0, 370.0],
            "density": [1.0, 1.1, 1.2],
        }
    ).to_csv(path, index=False)

    report = audit.audit_file(path)
    assert report["status"] == "ok"
    assert report["n_rows"] == 3
    assert report["proposed_representation_column"]["name"] == "psmiles"
    assert report["proposed_representation_column"]["confirmed"] is False
    assert report["proposed_tg_column"]["name"] == "Tg_K"
    assert report["proposed_tg_column"]["confirmed"] is False


def test_audit_reports_per_property_counts(tmp_path) -> None:
    """The Tg slice of a multi-property table must be measured, not inferred."""
    path = tmp_path / "sparse.csv"
    pd.DataFrame(
        {
            "psmiles": ["[*]CC[*]"] * 4,
            "Tg": [350.0, None, None, None],
            "density": [1.0, 1.1, 1.2, 1.3],
        }
    ).to_csv(path, index=False)

    counts = audit.audit_file(path)["per_numeric_column_non_null_counts"]
    assert counts["Tg"] == 1
    assert counts["density"] == 4


def test_ambiguous_tg_columns_are_not_auto_chosen(tmp_path) -> None:
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame(
        {
            "psmiles": ["[*]CC[*]", "[*]CCO[*]"],
            "Tg_dsc": [350.0, 360.0],
            "Tg_dma": [355.0, 365.0],
        }
    ).to_csv(path, index=False)

    report = audit.audit_file(path)
    assert report["proposed_tg_column"]["name"] is None
    assert "human must choose" in report["proposed_tg_column"]["note"]


def test_unreadable_file_is_reported_not_crashed(tmp_path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not a parquet file")
    report = audit.audit_file(path)
    assert report["status"] == "unreadable"


def test_audit_exits_2_when_confirmation_needed(tmp_path) -> None:
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({"psmiles": ["[*]CC[*]"], "Tg_a": [1.0], "Tg_b": [2.0]}).to_csv(
        path, index=False
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "AUDIT INCOMPLETE" in result.stderr
