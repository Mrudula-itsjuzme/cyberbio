from __future__ import annotations

from collections import defaultdict


class EmpiricalValidityModel:
    """Beta-Bernoulli proxy trained only from proposal validity labels."""

    def __init__(self, alpha=1.0, beta=1.0):
        self.prior_alpha = alpha; self.prior_beta = beta
        self.counts = defaultdict(lambda: [alpha, beta])

    def update(self, family: str, valid: bool) -> None:
        self.counts[family][0 if valid else 1] += 1.0

    def probability(self, family: str) -> float:
        alpha, beta = self.counts[family]
        return alpha / (alpha + beta)

    def snapshot(self):
        return {name: {"alpha": values[0], "beta": values[1], "p_valid": values[0] / sum(values)}
                for name, values in sorted(self.counts.items())}
