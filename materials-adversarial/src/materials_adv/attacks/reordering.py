"""Local rearrangement attack: swap adjacent editable tokens within a window.

Deliberately LOCAL. Arbitrary shuffling of a SMILES string is not a chemically
meaningful perturbation -- it destroys the structure rather than perturbing it,
and any resulting prediction drift would say nothing about model robustness.

This attack therefore swaps tokens only within a bounded window, preserves the
token multiset exactly, and never touches protected structural positions.
"""

from __future__ import annotations

import numpy as np

from .base import AttackOutcome, BaseAttack
from .registry import register_attack


@register_attack("reordering")
class ReorderingAttack(BaseAttack):
    def __init__(
        self,
        rng: np.random.Generator,
        *,
        window: int = 3,
        n_edits: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(rng, window=window, n_edits=n_edits, **kwargs)
        if window < 2:
            raise ValueError(f"window must be >= 2 for a swap to exist, got {window}")
        if n_edits < 1:
            raise ValueError(f"n_edits must be >= 1, got {n_edits}")
        self.window = window
        self.n_edits = n_edits

    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        editable = self._editable(tokens)
        if len(editable) < 2:
            return []

        # Candidate swaps: editable pairs within `window` of each other.
        pairs = [
            (a, b)
            for idx, a in enumerate(editable)
            for b in editable[idx + 1 :]
            if b - a <= self.window and tokens[a] != tokens[b]
        ]
        if not pairs:
            return []

        outcomes: list[AttackOutcome] = []
        seen: set[tuple[str, ...]] = set()
        for _ in range(n_variants):
            working = list(tokens)
            touched: list[int] = []
            for _ in range(self.n_edits):
                a, b = pairs[int(self.rng.integers(len(pairs)))]
                working[a], working[b] = working[b], working[a]
                touched.extend((a, b))
            adversarial = tuple(working)
            if adversarial in seen or adversarial == tuple(tokens):
                continue
            seen.add(adversarial)
            outcomes.append(
                AttackOutcome(
                    original_tokens=tuple(tokens),
                    adversarial_tokens=adversarial,
                    attack_type=self.name,
                    edit_positions=tuple(sorted(set(touched))),
                    params={"window": self.window, "n_edits": self.n_edits},
                )
            )
        return outcomes
