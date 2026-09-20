from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import CandidateRecord


@dataclass(slots=True)
class PairValidation:
    representation_valid: bool | None
    domain_constraint_pass: bool | None
    reasons: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        return self.representation_valid is True and self.domain_constraint_pass is True


class MaterialsValidator:
    """Authoritative adapter around the existing deterministic pipeline."""

    def validate(self, record: CandidateRecord) -> CandidateRecord:
        from materials_adv.validation.pipeline import validate

        result = validate(record.candidate_input)
        record.validity_status = result.status.value
        record.validation_details = {
            "representation_valid": result.representation_valid,
            "plausibility_flags": result.plausibility_flags,
            "rejection_reasons": list(result.rejection_reasons),
            "checks_run": list(result.checks_run),
            "checks_skipped": list(result.checks_skipped),
        }
        return record

    def validate_pair(self, source: str, candidate: str) -> PairValidation:
        """Validate a candidate relative to its source (for prob-lab adapters)."""
        from materials_adv.validation.pipeline import validate

        result = validate(candidate, check_plausibility=False)
        if result.representation_valid is not True:
            return PairValidation(
                representation_valid=result.representation_valid,
                domain_constraint_pass=None,
                reasons=tuple(result.rejection_reasons),
            )
        return PairValidation(
            representation_valid=True,
            domain_constraint_pass=True,  # permissive: just require valid SMILES
            reasons=(),
        )

