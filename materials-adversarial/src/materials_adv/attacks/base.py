"""Common attack interface.

Every attack implements `generate` (token list in, candidates out) and
`metadata` (its configuration, for the experiment record). New attacks are added
as one file plus a registry decoration, without touching the evaluator.

Attacks are deterministic given their seed: each takes an explicit
`numpy.random.Generator` rather than touching global RNG state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .token_space import count_changes


@dataclass(frozen=True, slots=True)
class AttackOutcome:
    """One candidate perturbation, before validation or model scoring.

    `number_of_changes` is computed from the token diff rather than reported by
    the attack itself, so a bookkeeping bug cannot understate perturbation size.
    """

    original_tokens: tuple[str, ...]
    adversarial_tokens: tuple[str, ...]
    attack_type: str
    edit_positions: tuple[int, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def number_of_changes(self) -> int:
        return count_changes(list(self.original_tokens), list(self.adversarial_tokens))

    @property
    def original_psmiles(self) -> str:
        return "".join(self.original_tokens)

    @property
    def adversarial_psmiles(self) -> str:
        return "".join(self.adversarial_tokens)

    @property
    def is_noop(self) -> bool:
        """True if the candidate is identical to the original.

        No-ops are retained rather than dropped: an attack that frequently fails
        to change anything is a reportable finding, not something to hide.
        """
        return self.original_tokens == self.adversarial_tokens


class BaseAttack(ABC):
    """Abstract attack.

    Subclasses set `name` and implement `generate`. `generate` MUST NOT mutate
    the token list it is given.
    """

    name: ClassVar[str] = "base"

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        protect_attachments: bool = True,
        protect_ring_closures: bool = True,
        protect_branches: bool = True,
        **params: Any,
    ) -> None:
        self.rng = rng
        self.protect_attachments = protect_attachments
        self.protect_ring_closures = protect_ring_closures
        self.protect_branches = protect_branches
        self.params = params

    @abstractmethod
    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        """Produce up to `n_variants` candidates. May return fewer, or none."""

    def metadata(self) -> dict[str, Any]:
        return {
            "attack_type": self.name,
            "protect_attachments": self.protect_attachments,
            "protect_ring_closures": self.protect_ring_closures,
            "protect_branches": self.protect_branches,
            **self.params,
        }

    def _editable(self, tokens: list[str]) -> tuple[int, ...]:
        from .token_space import editable_positions

        return editable_positions(
            tokens,
            protect_attachments=self.protect_attachments,
            protect_ring_closures=self.protect_ring_closures,
            protect_branches=self.protect_branches,
        )
