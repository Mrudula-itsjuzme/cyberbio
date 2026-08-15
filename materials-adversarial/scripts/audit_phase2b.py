"""Phase 2B: robustness comparison and scaler-confounder attribution.

Verifies the attack set is fresh (differs from Phase 1E) yet identical across
the two models being compared, then reports per-family drift statistics and the
attribution analysis separating adversarial augmentation from the scaler shift.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILIES = ["substitution", "insertion", "deletion", "rearrangement"]
THRESHOLD = 52.02  # Phase 1 baseline test MAE -- exploratory threshold, see limitations


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def by_family(records: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["validity_status"] == "valid" and r["prediction_drift"] is not None:
            out[r["attack_type"]].append(abs(r["prediction_drift"]))
    return out


def stats(drifts: list[float]) -> dict:
    a = np.asarray(drifts, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "rate_above_threshold": float((a > THRESHOLD).mean()),
        "n_above_threshold": int((a > THRESHOLD).sum()),
    }


def main() -> None:
    base = load("data/attacks/phase2b_baseline_results.jsonl")
    dfd = load("data/attacks/phase2b_defended_results.jsonl")
    p1e = load("data/attacks/phase1e_results.jsonl")

    base_str = [r["adversarial_psmiles"] for r in base]
    dfd_str = [r["adversarial_psmiles"] for r in dfd]
    p1e_set = set(r["adversarial_psmiles"] for r in p1e)

    integrity = {
        "n_candidates_baseline": len(base),
        "n_candidates_defended": len(dfd),
        "candidate_sets_identical": base_str == dfd_str,
        "n_unique_candidates": len(set(base_str)),
        "overlap_with_phase1e": len(set(base_str) & p1e_set),
        "overlap_fraction": len(set(base_str) & p1e_set) / len(set(base_str)),
    }

    print("=" * 78)
    print(f"{'PHASE 2B: CONTROLLED DEFENSE ABLATION':^78}")
    print("=" * 78)
    print("\n--- ATTACK SET INTEGRITY ---")
    print(f"  candidates (baseline / defended): {integrity['n_candidates_baseline']} / "
          f"{integrity['n_candidates_defended']}")
    print(f"  identical candidate sets (paired comparison): {integrity['candidate_sets_identical']}")
    print(f"  overlap with Phase 1E set: {integrity['overlap_with_phase1e']} / "
          f"{integrity['n_unique_candidates']} ({integrity['overlap_fraction']:.1%})")
    if not integrity["candidate_sets_identical"]:
        print("  WARNING: candidate sets differ -- comparison is NOT paired.")

    b_fam, d_fam = by_family(base), by_family(dfd)
    results = {"integrity": integrity, "threshold_K": THRESHOLD, "families": {}}

    print("\n--- PER-FAMILY DRIFT (valid candidates only) ---")
    hdr = f"{'family':<15} {'n':>4} {'mean':>16} {'median':>16} {'P95':>16} {'max':>16} {'>52.02K':>16}"
    print(hdr)
    print("-" * len(hdr))

    for fam in FAMILIES:
        if fam not in b_fam or fam not in d_fam:
            continue
        b, d = stats(b_fam[fam]), stats(d_fam[fam])
        results["families"][fam] = {"baseline": b, "phase2b_controlled": d}

        def cell(bv: float, dv: float, pct: bool = False) -> str:
            if pct:
                return f"{bv:.1%}->{dv:.1%}"
            return f"{bv:.1f}->{dv:.1f}"

        print(
            f"{fam:<15} {b['n']:>4} "
            f"{cell(b['mean'], d['mean']):>16} "
            f"{cell(b['median'], d['median']):>16} "
            f"{cell(b['p95'], d['p95']):>16} "
            f"{cell(b['max'], d['max']):>16} "
            f"{cell(b['rate_above_threshold'], d['rate_above_threshold'], True):>16}"
        )

    # Pooled paired comparison over identical candidates.
    paired = [
        (abs(rb["prediction_drift"]), abs(rd["prediction_drift"]))
        for rb, rd in zip(base, dfd)
        if rb["validity_status"] == "valid"
        and rb["prediction_drift"] is not None
        and rd["prediction_drift"] is not None
    ]
    if paired:
        pb = np.array([p[0] for p in paired])
        pd_ = np.array([p[1] for p in paired])
        diff = pb - pd_
        results["paired_pooled"] = {
            "n": len(paired),
            "baseline_mean": float(pb.mean()),
            "defended_mean": float(pd_.mean()),
            "mean_reduction": float(diff.mean()),
            "median_reduction": float(np.median(diff)),
            "fraction_improved": float((diff > 0).mean()),
        }
        print("\n--- POOLED PAIRED COMPARISON (same candidate, both models) ---")
        print(f"  n paired valid candidates : {len(paired)}")
        print(f"  mean |drift| baseline     : {pb.mean():.2f} K")
        print(f"  mean |drift| Phase 2B     : {pd_.mean():.2f} K")
        print(f"  mean reduction            : {diff.mean():.2f} K")
        print(f"  candidates improved       : {(diff > 0).mean():.1%}")

    # Clean-metric attribution.
    clean = json.loads(Path("results/phase2b_clean_eval.json").read_text())
    p1 = clean["phase1_baseline"]["test"]
    p2a = clean["phase2a_defended"]["test"]
    p2b = clean["phase2b_controlled"]["test"]
    swap = clean.get("phase2a_decoded_with_phase1_scaler", {}).get("test")

    total = p1["mae"] - p2a["mae"]
    controlled = p1["mae"] - p2b["mae"]
    results["attribution"] = {
        "phase1_test_mae": p1["mae"],
        "phase2a_test_mae": p2a["mae"],
        "phase2b_test_mae": p2b["mae"],
        "total_apparent_improvement_2a": total,
        "improvement_surviving_control_2b": controlled,
        "residual_associated_with_scaler_change": total - controlled,
        "fraction_surviving_control": controlled / total if total else float("nan"),
    }

    print("\n--- CLEAN TEST MAE ATTRIBUTION (n=6, see limitations) ---")
    print(f"  Phase 1 baseline                 : {p1['mae']:.2f} K")
    print(f"  Phase 2A (augmentation + refit)  : {p2a['mae']:.2f} K   "
          f"(apparent improvement {total:+.2f} K)")
    print(f"  Phase 2B (augmentation + fixed)  : {p2b['mae']:.2f} K   "
          f"(controlled improvement {controlled:+.2f} K)")
    if swap:
        print(f"  [diagnostic] 2A model, P1 scaler : {swap['mae']:.2f} K")
    print(f"\n  Improvement surviving the control: {controlled:.2f} K of {total:.2f} K "
          f"({controlled / total:.0%})" if total else "")
    print(f"  Residual associated with scaler  : {total - controlled:.2f} K")

    out = Path("results/phase2b_audit.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
