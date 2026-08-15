#!/usr/bin/env python3
"""Dataset audit: DISCOVER the schema, never assume it.

Read-only. Proposes candidate columns with evidence and `confirmed: false`, and
never auto-writes configs/dataset.yaml. A human confirms the representation
column, the Tg column and the unit verdict before any model code runs.

Usage:
    python scripts/audit_dataset.py --path data/raw
    python scripts/audit_dataset.py --path data/raw/file.csv --output results/audit.json

Exit codes:
    0  audit completed, no blocking ambiguity
    1  no readable data found
    2  audit completed but requires human confirmation before proceeding
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.data.tokenizer import (  # noqa: E402
    TOKEN_PATTERN,
    unknown_character_histogram,
)

KELVIN_CELSIUS_OFFSET = 273.15
TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json"}
SAMPLE_ROWS = 5000


# --------------------------------------------------------------------------
# File discovery and loading
# --------------------------------------------------------------------------
def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_SUFFIXES
    )


def load_table(path: Path, nrows: int | None = None) -> tuple[pd.DataFrame | None, str]:
    """Load a table, returning (df, note). Never raises on a bad file."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path, nrows=nrows, sep=None, engine="python"), "ok"
        if suffix == ".tsv":
            return pd.read_csv(path, nrows=nrows, sep="\t"), "ok"
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=nrows), "ok"
        if suffix == ".parquet":
            try:
                return pd.read_parquet(path), "ok"
            except ImportError:
                return None, "cannot inspect: parquet engine (pyarrow) not installed"
        if suffix == ".json":
            return pd.read_json(path), "ok"
    except Exception as exc:
        return None, f"failed to read: {type(exc).__name__}: {exc}"
    return None, f"unsupported suffix {suffix}"


# --------------------------------------------------------------------------
# Column scoring -- evidence, not decisions
# --------------------------------------------------------------------------
def _is_text_dtype(series: pd.Series) -> bool:
    """True for string-like columns.

    NOT `dtype == object`: pandas 3.0 assigns string columns a dedicated 'str'
    dtype, so the object check silently finds nothing and the audit would report
    "no representation column" for a perfectly good file. Both conventions are
    accepted so the script works on pandas 2.x and 3.x alike.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    return (
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
        or str(series.dtype) == "str"
    )



def score_representation_column(series: pd.Series) -> dict[str, Any]:
    """Score how much a column looks like PSMILES.

    Uses the project's own tokenizer rather than column-name matching: real
    PSMILES tokenizes cleanly and contains attachment points, and that evidence
    is far more robust than hoping for a column called 'smiles'.
    """
    values = series.dropna().astype(str)
    values = values[values.str.strip() != ""]
    if values.empty:
        return {"score": 0.0, "reason": "empty column"}

    sample = values.head(500).tolist()
    n = len(sample)

    fully_tokenizable = 0
    with_attachment = 0
    for text in sample:
        i, ok = 0, True
        while i < len(text):
            m = TOKEN_PATTERN.match(text, i)
            if m is None:
                ok = False
                break
            i = m.end()
        if ok:
            fully_tokenizable += 1
        if "*" in text:
            with_attachment += 1

    frac_tokenizable = fully_tokenizable / n
    frac_attachment = with_attachment / n
    lengths = values.str.len()
    balanced = (values.str.count(r"\(") == values.str.count(r"\)")).mean()

    # Attachment points are the strongest polymer-specific signal.
    score = 0.5 * frac_tokenizable + 0.35 * frac_attachment + 0.15 * float(balanced)

    return {
        "score": round(float(score), 4),
        "frac_fully_tokenizable": round(frac_tokenizable, 4),
        "frac_containing_attachment_star": round(frac_attachment, 4),
        "frac_balanced_parens": round(float(balanced), 4),
        "length_min": int(lengths.min()),
        "length_median": float(lengths.median()),
        "length_max": int(lengths.max()),
        "n_sampled": n,
        "unknown_char_histogram": dict(
            Counter(unknown_character_histogram(sample)).most_common(10)
        ),
    }


def analyze_tg_units(values: np.ndarray) -> dict[str, Any]:
    """Evidence about Kelvin vs Celsius. Never auto-converts.

    Method note: a naive largest-gap bimodality probe was tested against a
    synthetic 50/50 C+K mixture and FAILED -- it flagged a spurious gap in the
    far-left tail because the two components overlap. This uses instead:

      (a) negative values, which conclusively RULE OUT pure Kelvin
      (b) a shift-correlation test: correlate the histogram with itself shifted
          by 273.15 and compare against the zero-shift baseline
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"verdict": "undetermined", "reason": "no finite values"}

    n_negative = int((values < 0).sum())
    vmin, vmax = float(values.min()), float(values.max())

    evidence: dict[str, Any] = {
        "n": int(values.size),
        "min": vmin,
        "max": vmax,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "n_negative": n_negative,
        "n_above_400": int((values > 400).sum()),
        "n_below_150": int((values < 150).sum()),
    }

    # Shift-correlation: a genuine K/C mixture shows self-similarity at ~273.15.
    shift_ratio = None
    if values.size >= 30 and (vmax - vmin) > KELVIN_CELSIUS_OFFSET:
        bins = np.arange(vmin, vmax + 5.0, 5.0)
        hist, _ = np.histogram(values, bins=bins)
        hist = hist.astype(float)
        shift_bins = int(round(KELVIN_CELSIUS_OFFSET / 5.0))
        if 0 < shift_bins < len(hist):
            baseline = float(np.dot(hist, hist))
            shifted = float(np.dot(hist[:-shift_bins], hist[shift_bins:]))
            if baseline > 0:
                shift_ratio = round(shifted / baseline, 4)
    evidence["shift_correlation_ratio_at_273K"] = shift_ratio

    if n_negative > 0:
        verdict = "likely_celsius" if evidence["n_above_400"] == 0 else "possibly_mixed"
        reason = f"{n_negative} negative value(s) rule out pure Kelvin"
    elif vmin > 150 and evidence["n_below_150"] == 0:
        verdict = "likely_kelvin"
        reason = "all values above 150 with none negative; consistent with Kelvin"
    elif shift_ratio is not None and shift_ratio > 0.30:
        verdict = "possibly_mixed"
        reason = f"histogram self-similarity at +273.15 (ratio {shift_ratio})"
    else:
        verdict = "undetermined"
        reason = "range is consistent with more than one unit convention"

    evidence["verdict"] = verdict
    evidence["reason"] = reason
    evidence["requires_human_confirmation"] = True
    return evidence


def find_duplicate_conflicts(
    df: pd.DataFrame, rep_col: str, prop_col: str, tolerance: float = 1e-6
) -> dict[str, Any]:
    """Split duplicates into exact / conflicting / near-duplicate. Never resolves them."""
    sub = df[[rep_col, prop_col]].dropna()
    if sub.empty:
        return {"note": "no rows with both representation and property"}

    grouped = sub.groupby(rep_col)[prop_col]
    spread = grouped.max() - grouped.min()
    counts = grouped.size()

    duplicated = counts[counts > 1]
    conflicting = spread[(counts > 1) & (spread > tolerance)]

    examples = []
    for rep in conflicting.sort_values(ascending=False).head(5).index:
        vals = sorted(sub.loc[sub[rep_col] == rep, prop_col].tolist())
        near_273 = any(
            abs(abs(a - b) - KELVIN_CELSIUS_OFFSET) < 1.0
            for a in vals
            for b in vals
            if a != b
        )
        examples.append(
            {
                "representation": str(rep)[:120],
                "values": vals,
                "spread": float(max(vals) - min(vals)),
                "differs_by_approx_273.15": near_273,
            }
        )

    normalized = sub[rep_col].astype(str).str.strip().str.lower()
    return {
        "n_rows": int(len(sub)),
        "n_unique_representations": int(sub[rep_col].nunique()),
        "n_representations_appearing_more_than_once": int(len(duplicated)),
        "n_conflicting_representations": int(len(conflicting)),
        "max_conflict_spread": float(conflicting.max()) if len(conflicting) else 0.0,
        "n_near_duplicates_after_normalization": int(
            sub[rep_col].nunique() - normalized.nunique()
        ),
        "conflict_examples": examples,
        "resolution_policy": "NOT AUTO-RESOLVED -- requires an explicit human decision",
    }


# --------------------------------------------------------------------------
# Audit driver
# --------------------------------------------------------------------------
def audit_file(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"file": str(path), "size_bytes": path.stat().st_size}

    df, note = load_table(path, nrows=SAMPLE_ROWS)
    if df is None:
        report["status"] = "unreadable"
        report["note"] = note
        return report

    full_df, full_note = load_table(path)
    if full_df is not None:
        df = full_df
    else:
        report["note"] = f"full read failed ({full_note}); audited a {SAMPLE_ROWS}-row sample"

    report["status"] = "ok"
    report["n_rows"] = int(len(df))
    report["n_columns"] = int(len(df.columns))
    report["columns"] = [
        {
            "name": str(c),
            "dtype": str(df[c].dtype),
            "n_non_null": int(df[c].notna().sum()),
            "n_null": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True)),
        }
        for c in df.columns
    ]

    # Per-property non-null counts: OpenPoly spreads ~3985 pairs over 26
    # properties, so the usable Tg slice must be measured, never inferred.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    report["per_numeric_column_non_null_counts"] = {
        str(c): int(df[c].notna().sum()) for c in numeric_cols
    }

    text_cols = [c for c in df.columns if _is_text_dtype(df[c])]
    rep_scores = {str(c): score_representation_column(df[c]) for c in text_cols}
    report["representation_column_candidates"] = dict(
        sorted(rep_scores.items(), key=lambda kv: -kv[1].get("score", 0))
    )

    best_rep = None
    if rep_scores:
        best_name, best = max(rep_scores.items(), key=lambda kv: kv[1].get("score", 0))
        if best.get("score", 0) >= 0.5:
            best_rep = best_name
    report["proposed_representation_column"] = {
        "name": best_rep,
        "confirmed": False,
        "note": "PROPOSAL ONLY -- confirm before use",
    }

    # Tg candidates by name hint, scored on evidence not names alone.
    tg_hints = ("tg", "glass", "transition")
    tg_candidates = [
        c for c in numeric_cols if any(h in str(c).lower() for h in tg_hints)
    ]
    report["tg_column_candidates"] = {
        str(c): {
            "n_non_null": int(df[c].notna().sum()),
            "units": analyze_tg_units(df[c].dropna().to_numpy(dtype=float)),
        }
        for c in tg_candidates
    }
    report["proposed_tg_column"] = {
        "name": str(tg_candidates[0]) if len(tg_candidates) == 1 else None,
        "confirmed": False,
        "note": (
            "PROPOSAL ONLY"
            if len(tg_candidates) == 1
            else f"{len(tg_candidates)} candidates -- human must choose"
        ),
    }

    if best_rep and len(tg_candidates) == 1:
        report["duplicate_analysis"] = find_duplicate_conflicts(
            df, best_rep, tg_candidates[0]
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", required=True, help="data file or directory")
    parser.add_argument("--output", default=None, help="write JSON report here")
    args = parser.parse_args()

    root = Path(args.path)
    files = discover_files(root)

    if not files:
        print(f"BLOCKED: no readable tabular data found at {root}", file=sys.stderr)
        print(
            "\nThe OpenPoly dataset is required. Source:\n"
            "  Paper : Wang et al., 'OpenPoly: A Polymer Database Empowering Benchmarking\n"
            "          and Multi-property Predictions', Chinese Journal of Polymer Science\n"
            "          (2025). DOI 10.1007/s10118-025-3402-y\n"
            "  Data  : https://github.com/WangGroupFDU/Openpoly_benchmark\n"
            "          data/final_polymer_properties_fromliterature.csv\n"
            f"\nPlace the file(s) in {root}/ and re-run.\n"
            "\nNo data will be fabricated and no model will be trained until this is resolved.",
            file=sys.stderr,
        )
        return 1

    report: dict[str, Any] = {
        "audited_path": str(root),
        "n_files": len(files),
        "files": [audit_file(f) for f in files],
    }

    text = json.dumps(report, indent=2, default=str)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"Report written to {out}")
    else:
        print(text)

    needs_confirmation = any(
        f.get("status") == "ok"
        and (
            not f.get("proposed_representation_column", {}).get("name")
            or not f.get("proposed_tg_column", {}).get("name")
        )
        for f in report["files"]
    )
    if needs_confirmation:
        print(
            "\nAUDIT INCOMPLETE: representation and/or Tg column could not be "
            "unambiguously proposed. Human confirmation required before proceeding.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
