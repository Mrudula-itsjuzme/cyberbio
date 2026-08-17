"""Substitution attack: replace an allowed token with another allowed token.

MECHANICS ARE COMPLETE. The replacement POOL is PENDING(dataset).

Why the pool cannot be hardcoded
--------------------------------
Deletion, insertion and reordering are *closed* operations -- they need only
tokens already present in the sequence. Substitution is *open*: it must answer
"replace with what?"

Hardcoding a replacement list (say, C/N/O/S) would inject the author's chemical
priors directly into the headline result. Worse, if the replacement distribution
does not reflect the training corpus, the attack measures out-of-distribution
handling rather than adversarial robustness -- a different scientific claim,
easily mistaken for the intended one.

`allowed_tokens` is therefore a REQUIRED injected argument. It should be derived
from the training-split vocabulary once the dataset exists. The class is fully
testable now by injecting a synthetic pool.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..utils.pending import PendingImplementation
from .base import AttackOutcome, BaseAttack
from .registry import register_attack
from .token_space import TokenRole, classify_token


@register_attack("substitution")
class SubstitutionAttack(BaseAttack):
    def __init__(
        self,
        rng: np.random.Generator,
        *,
        allowed_tokens: Sequence[str] | None = None,
        attack_budget: int = 1,
        role_preserving: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(rng, attack_budget=attack_budget, role_preserving=role_preserving, **kwargs)
        if allowed_tokens is None:
            raise PendingImplementation(
                what=(
                    "SubstitutionAttack requires an explicit `allowed_tokens` pool. "
                    "Mechanics are implemented; the pool is deliberately not hardcoded "
                    "because a hand-written replacement set would encode the author's "
                    "chemical priors into the results."
                ),
                blocked_on="dataset",
                unblocks_when=(
                    "the training-split vocabulary exists; derive the pool from it via "
                    "Vocabulary.build(train_texts) and pass allowed_tokens=..."
                ),
            )
        if attack_budget < 1:
            raise ValueError(f"attack_budget must be >= 1, got {attack_budget}")
        self.allowed_tokens = tuple(allowed_tokens)
        self.attack_budget = attack_budget
        self.role_preserving = role_preserving

    def _candidates_for(self, token: str) -> tuple[str, ...]:
        """Replacements for `token`, excluding itself.

        With `role_preserving` (default), an atom is only replaced by an atom.
        Swapping an atom for a bond symbol almost always yields an invalid
        string, which would confound drift measurement with filter behaviour.
        """
        pool = [t for t in self.allowed_tokens if t != token]
        if not self.role_preserving:
            return tuple(pool)
        role = classify_token(token)
        same_role = [t for t in pool if classify_token(t) is role]
        return tuple(same_role)

    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        editable = self._editable(tokens)
        # Structural roles are excluded even when unprotected: substituting a
        # branch paren or ring digit is a guaranteed-invalidity edit.
        excluded = {TokenRole.BRANCH_OPEN, TokenRole.BRANCH_CLOSE, TokenRole.RING_CLOSURE}
        sites = [
            i
            for i in editable
            if classify_token(tokens[i]) not in excluded and self._candidates_for(tokens[i])
        ]
        if len(sites) < self.attack_budget:
            return []

        outcomes: list[AttackOutcome] = []
        seen: set[tuple[str, ...]] = set()
        for _ in range(n_variants):
            working = list(tokens)
            chosen = self.rng.choice(len(sites), size=self.attack_budget, replace=False)
            positions = sorted(sites[int(c)] for c in chosen)
            for pos in positions:
                candidates = self._candidates_for(tokens[pos])
                if not candidates:
                    continue
                working[pos] = str(self.rng.choice(candidates))
            adversarial = tuple(working)
            if adversarial in seen or adversarial == tuple(tokens):
                continue
            seen.add(adversarial)
            outcomes.append(
                AttackOutcome(
                    original_tokens=tuple(tokens),
                    adversarial_tokens=adversarial,
                    attack_type=self.name,
                    edit_positions=tuple(positions),
                    params={
                        "attack_budget": self.attack_budget,
                        "role_preserving": self.role_preserving,
                        "pool_size": len(self.allowed_tokens),
                    },
                )
            )
        return outcomes
