"""ValidationResult and the combined validity pipeline.

The three concepts the project requires be kept separate:

  1. REPRESENTATION VALIDITY  -- is the string well-formed and parseable?
  2. CHEMICAL PLAUSIBILITY    -- is it a sensible polymer repeat unit?
  3. ADVERSARIAL EFFECTIVENESS -- does it move the model's prediction?

(3) lives in evaluation/attack_metrics.py. (1) and (2) are separate modules and
are never collapsed into one boolean, because "RDKit parsed it" is not evidence
of scientific plausibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..evaluation.records import ValidityStatus


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Structured validity information.

    `representation_valid` is tri-state and the distinction is load-bearing:
        True  -- verified parseable
        False -- verified unparseable
        None  -- NOT CHECKED (e.g. RDKit unavailable)

    None must never be collapsed to False. "We did not check" and "it failed" are
    different claims, and conflating them would silently inflate reported
    rejection rates.
    """

    representation_valid: bool | None
    plausibility_flags: dict[str, bool] = field(default_factory=dict)
    rejection_reasons: tuple[str, ...] = ()
    checks_run: tuple[str, ...] = ()
    checks_skipped: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        raise TypeError(
            "ValidationResult has no truth value -- `if result:` would always be "
            "True and silently hide invalid candidates. Inspect .status, "
            ".representation_valid, or .plausible explicitly."
        )

    @property
    def plausible(self) -> bool:
        """True only if every plausibility flag passed. Empty flags => not asserted."""
        return bool(self.plausibility_flags) and all(self.plausibility_flags.values())

    @property
    def status(self) -> ValidityStatus:
        if self.representation_valid is False:
            return ValidityStatus.INVALID_REPRESENTATION
        if self.representation_valid is None:
            return ValidityStatus.UNCHECKED
        if self.plausibility_flags and not self.plausible:
            return ValidityStatus.IMPLAUSIBLE
        return ValidityStatus.VALID


def validate(psmiles: str, *, check_plausibility: bool = True) -> ValidationResult:
    """Run representation validity, then plausibility heuristics.

    Plausibility is only meaningful on a well-formed string, so it is skipped
    (and reported as skipped) when representation validity fails.
    """
    from .plausibility import check_plausibility as _check_plausibility
    from .representation import validate_representation

    rep = validate_representation(psmiles)

    if rep.representation_valid is False or not check_plausibility:
        skipped = tuple(rep.checks_skipped) + ("plausibility",)
        return ValidationResult(
            representation_valid=rep.representation_valid,
            plausibility_flags=rep.plausibility_flags,
            rejection_reasons=rep.rejection_reasons,
            checks_run=rep.checks_run,
            checks_skipped=skipped,
        )

    flags, reasons = _check_plausibility(psmiles)
    return ValidationResult(
        representation_valid=rep.representation_valid,
        plausibility_flags={**rep.plausibility_flags, **flags},
        rejection_reasons=rep.rejection_reasons + tuple(reasons),
        checks_run=rep.checks_run + ("plausibility",),
        checks_skipped=rep.checks_skipped,
    )
