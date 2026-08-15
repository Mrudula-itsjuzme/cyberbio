"""Attack evaluation metrics.

Two rules the project requires:

1. Attack success is NOT "the prediction changed". A configurable threshold
   combines validity, perturbation size and drift magnitude, so the definition
   stays a research parameter rather than a hardcoded assumption.

2. Rejected, invalid and UNCHECKED candidates are counted and reported. Summaries
   that quietly average over survivors only would overstate attack effectiveness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .records import AttackRecord, ValidityStatus


@dataclass(frozen=True, slots=True)
class SuccessCriterion:
    """Configurable definition of a successful attack.

    `require_valid` defaults to True. Note that under the strict-UNCHECKED policy
    an UNCHECKED candidate is NOT valid, so with RDKit absent this criterion will
    report zero successes -- which is the honest outcome, not a bug. Set
    `count_unchecked_as_valid=True` only for an explicitly-labelled syntax-only
    analysis.
    """

    min_abs_drift: float = 0.0
    max_changes: int | None = None
    require_valid: bool = True
    count_unchecked_as_valid: bool = False

    def is_success(self, record: AttackRecord) -> bool:
        if record.prediction_drift is None:
            return False
        if abs(record.prediction_drift) < self.min_abs_drift:
            return False
        if self.max_changes is not None and record.number_of_changes > self.max_changes:
            return False
        if self.require_valid:
            status = record.validity_status
            if status == ValidityStatus.VALID.value:
                return True
            if (
                self.count_unchecked_as_valid
                and status == ValidityStatus.UNCHECKED.value
            ):
                return True
            return False
        return True


def summarize_attacks(
    records: Sequence[AttackRecord],
    criterion: SuccessCriterion | None = None,
) -> dict[str, object]:
    """Aggregate attack records, reporting failures alongside successes."""
    criterion = criterion or SuccessCriterion()
    n_total = len(records)
    if n_total == 0:
        return {"n_total": 0, "note": "no records"}

    status_counts: dict[str, int] = {}
    for r in records:
        status_counts[r.validity_status] = status_counts.get(r.validity_status, 0) + 1

    scored = [r for r in records if r.prediction_drift is not None]
    drifts = np.array([r.prediction_drift for r in scored], dtype=float)
    abs_drifts = np.abs(drifts)
    changes = np.array([r.number_of_changes for r in records], dtype=float)
    successes = [r for r in records if criterion.is_success(r)]

    summary: dict[str, object] = {
        "n_total": n_total,
        "n_scored": len(scored),
        "n_unscored": n_total - len(scored),
        "n_noop": sum(1 for r in records if r.original_psmiles == r.adversarial_psmiles),
        "status_counts": status_counts,
        "n_unchecked": status_counts.get(ValidityStatus.UNCHECKED.value, 0),
        "n_invalid": status_counts.get(ValidityStatus.INVALID_REPRESENTATION.value, 0),
        "n_implausible": status_counts.get(ValidityStatus.IMPLAUSIBLE.value, 0),
        "n_success": len(successes),
        "success_rate": len(successes) / n_total,
        "criterion": {
            "min_abs_drift": criterion.min_abs_drift,
            "max_changes": criterion.max_changes,
            "require_valid": criterion.require_valid,
            "count_unchecked_as_valid": criterion.count_unchecked_as_valid,
        },
        "mean_changes": float(np.mean(changes)) if changes.size else None,
    }

    if abs_drifts.size:
        summary.update(
            {
                "mean_abs_drift": float(np.mean(abs_drifts)),
                "median_abs_drift": float(np.median(abs_drifts)),
                "max_abs_drift": float(np.max(abs_drifts)),
                # Signed mean reveals directional bias that abs() would hide.
                "mean_signed_drift": float(np.mean(drifts)),
            }
        )
    return summary


def group_by_attack_type(
    records: Sequence[AttackRecord],
    criterion: SuccessCriterion | None = None,
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[AttackRecord]] = {}
    for r in records:
        groups.setdefault(r.attack_type, []).append(r)
    return {k: summarize_attacks(v, criterion) for k, v in sorted(groups.items())}
