"""Attack record schema.

Phase 2+ fields (attack_score, uncertainty_change, ...) exist NOW and are emitted
as null. Adding them later would force an on-disk schema migration mid-project
and break comparability between Phase 1 and Phase 2 results.

`prediction_drift` is stored SIGNED. Absolute value discards direction, and
"does this attack push Tg up or down" is a real research question. abs() is
applied at reporting time, in evaluation/attack_metrics.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.1"

class ValidityStatus(str, Enum):
    VALID = "valid"
    INVALID_REPRESENTATION = "invalid_representation"
    IMPLAUSIBLE = "implausible"
    UNCHECKED = "unchecked"

@dataclass(frozen=True, slots=True)
class AttackRecord:
    attack_id: str
    sample_id: str
    original_representation: str
    adversarial_representation: str
    attack_type: str
    attack_budget: int
    number_of_changes: int
    edited_positions: tuple[int, ...]
    validity_status: str
    plausibility_status: str

    original_prediction: float | None = None
    adversarial_prediction: float | None = None
    signed_prediction_drift: float | None = None
    absolute_prediction_drift: float | None = None
    attack_success: bool = False

    rejection_reasons: tuple[str, ...] = ()
    plausibility_flags: dict[str, bool] = field(default_factory=dict)
    checks_skipped: tuple[str, ...] = ()

    # --- Reserved for later phases; emitted as null in Phase 1 ---
    attack_score: float | None = None
    confidence_change: float | None = None
    uncertainty_change: float | None = None
    chemical_constraint_score: float | None = None
    sampling_probability: float | None = None

    # --- Provenance ---
    seed: int | None = None
    attack_params: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rejection_reasons"] = list(self.rejection_reasons)
        d["checks_skipped"] = list(self.checks_skipped)
        d["edited_positions"] = [int(p) for p in self.edited_positions]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttackRecord:
        d = dict(d)
        d["rejection_reasons"] = tuple(d.get("rejection_reasons", ()))
        d["checks_skipped"] = tuple(d.get("checks_skipped", ()))
        d["edited_positions"] = tuple(d.get("edited_positions", ()))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def make_attack_id(sample_id: str, attack_type: str, seed: int | None, ordinal: int) -> str:
    """Deterministic attack id, so a rerun with the same seed reproduces ids."""
    return f"{sample_id}:{attack_type}:{seed if seed is not None else 'noseed'}:{ordinal}"


def compute_drift(original: float | None, adversarial: float | None) -> float | None:
    """Signed drift. None if either prediction is missing."""
    if original is None or adversarial is None:
        return None
    return float(adversarial) - float(original)

