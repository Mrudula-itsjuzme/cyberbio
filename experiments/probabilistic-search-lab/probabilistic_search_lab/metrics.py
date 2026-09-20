from __future__ import annotations

import math
from statistics import fmean, median


def summarize_effects(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "best": None}
    ordered = sorted(values)
    index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "mean": fmean(ordered),
        "median": median(ordered),
        "p90": ordered[index],
        "best": ordered[-1],
    }


def categorical_entropy(probabilities: list[float]) -> float:
    return -sum(p * math.log(p) for p in probabilities if p > 0)
