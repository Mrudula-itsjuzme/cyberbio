from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class Validation:
    representation_valid: bool | None
    domain_constraint_pass: bool | None
    reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return self.representation_valid is True and self.domain_constraint_pass is True


@dataclass(frozen=True, slots=True)
class Proposal:
    candidate: str
    family: str
    edit_cost: float = 1.0
    trajectory: tuple[str, ...] = ()


class TargetModel(Protocol):
    def predict(self, sequence: str) -> float: ...


class Validator(Protocol):
    def validate(self, source: str, candidate: str) -> Validation: ...


class ProposalFamily(Protocol):
    name: str

    def propose(self, source: str, current: str, rng) -> Sequence[Proposal]: ...
