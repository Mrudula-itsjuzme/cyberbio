"""Proposal operators shared by all Phase 6 search policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..base import AttackOutcome, BaseAttack


@dataclass(frozen=True)
class Proposal:
    tokens: tuple[str, ...]
    operator: str
    outcome: AttackOutcome


class CompositeProposalOperator:
    """Uniformly select one existing mutation operator and request one edit."""

    def __init__(self, operators: Sequence[BaseAttack], rng: np.random.Generator):
        if not operators:
            raise ValueError("at least one proposal operator is required")
        self.operators = tuple(operators)
        self.rng = rng

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(operator.name for operator in self.operators)

    def propose(self, tokens: Sequence[str]) -> Proposal | None:
        operator = self.operators[int(self.rng.integers(0, len(self.operators)))]
        outcomes = operator.generate(list(tokens), n_variants=1)
        if not outcomes:
            return None
        outcome = outcomes[0]
        return Proposal(tuple(outcome.adversarial_tokens), operator.name, outcome)

    def enumerate_proposals(self, tokens: Sequence[str]) -> list[Proposal]:
        proposals = []
        for operator in self.operators:
            try:
                outcomes = operator.enumerate(list(tokens))
                for outcome in outcomes:
                    proposals.append(Proposal(tuple(outcome.adversarial_tokens), operator.name, outcome))
            except NotImplementedError:
                continue
        return proposals
