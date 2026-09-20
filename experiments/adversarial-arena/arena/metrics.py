from __future__ import annotations

from collections import Counter
import math
import numpy as np

from .contracts import CandidateRecord


def attack_metrics(records: list[CandidateRecord]) -> dict:
    valid = [r for r in records if r.validity_status == "valid"]
    drifts = np.asarray([r.drift for r in valid if r.drift is not None], dtype=float)
    queries = sum(r.target_query_count for r in records)
    unique = len({r.candidate_input for r in valid})
    return {
        "proposals": len(records), "valid": len(valid), "invalid": len(records) - len(valid),
        "validity_rate": len(valid) / len(records) if records else None,
        "mean_drift": float(drifts.mean()) if drifts.size else None,
        "median_drift": float(np.median(drifts)) if drifts.size else None,
        "p90_drift": float(np.quantile(drifts, .9)) if drifts.size else None,
        "max_drift": float(drifts.max()) if drifts.size else None,
        "queries": queries, "drift_per_query": float(drifts.sum() / queries) if queries else None,
        "proposal_efficiency": len(valid) / len(records) if records else None,
        "candidate_diversity": unique / len(valid) if valid else None,
        "family_distribution": dict(Counter(r.attack_family for r in records)),
    }


def gain(before: float | None, after: float | None) -> float | None:
    return None if before is None or after in (None, 0) else float(before / after)


def policy_entropy(probabilities: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in probabilities.values() if p > 0)

