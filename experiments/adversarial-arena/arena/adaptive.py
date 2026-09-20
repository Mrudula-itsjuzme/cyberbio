from __future__ import annotations

import numpy as np


class ThompsonController:
    """Beta Thompson controller using normalized drift/query as Bernoulli evidence."""

    def __init__(self, arms: list[str], seed: int):
        self.arms = list(arms)
        self.rng = np.random.default_rng(seed)
        self.alpha = {arm: 1.0 for arm in arms}
        self.beta = {arm: 1.0 for arm in arms}
        self.history: list[dict] = []

    def select(self) -> str:
        draws = {a: float(self.rng.beta(self.alpha[a], self.beta[a])) for a in self.arms}
        arm = max(self.arms, key=lambda a: (draws[a], a))
        self.history.append({"selected": arm, "draws": draws})
        return arm

    def update(self, arm: str, reward: float) -> None:
        reward = min(1.0, max(0.0, float(reward)))
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    def state(self) -> dict:
        means = {a: self.alpha[a] / (self.alpha[a] + self.beta[a]) for a in self.arms}
        return {"alpha": self.alpha, "beta": self.beta, "posterior_means": means, "history": self.history}

