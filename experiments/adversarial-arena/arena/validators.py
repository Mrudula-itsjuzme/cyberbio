from __future__ import annotations

from .contracts import CandidateRecord


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

