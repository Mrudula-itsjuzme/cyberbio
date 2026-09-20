from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass(slots=True)
class SourceSample:
    source_id: str
    input: str
    target: float | None = None


@dataclass(slots=True)
class CandidateRecord:
    source_id: str
    original_input: str
    candidate_input: str
    attack_family: str
    attack_variant: str
    proposal_step: int
    validity_status: str = "unchecked"
    validation_details: dict[str, Any] = field(default_factory=dict)
    target_query_count: int = 0
    score: float | None = None
    clean_prediction: float | None = None
    challenging_prediction: float | None = None
    drift: float | None = None
    true_target: float | None = None
    edit_distance: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Attacker(ABC):
    name = "base"
    version = "1"

    @abstractmethod
    def propose(self, samples: Sequence[SourceSample], defender: "Defender") -> list[CandidateRecord]: ...

    def update(self, rewards: dict[str, float]) -> None:  # static default
        return None

    @abstractmethod
    def reset(self, seed: int) -> None: ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]: ...

    def get_metrics(self) -> dict[str, Any]:
        return {}


class Defender(ABC):
    round_id: int
    checkpoint_id: str

    @abstractmethod
    def predict(self, inputs: Sequence[str]) -> list[float]: ...

    @abstractmethod
    def train(self, clean: Sequence[SourceSample], hard_examples: Sequence[CandidateRecord], *, round_id: int) -> dict[str, Any]: ...

    @abstractmethod
    def evaluate(self, samples: Sequence[SourceSample]) -> dict[str, float | int | None]: ...

    @abstractmethod
    def checkpoint(self, path: str) -> str: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]: ...

