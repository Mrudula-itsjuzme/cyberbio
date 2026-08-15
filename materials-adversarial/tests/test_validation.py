"""Validation tests, with emphasis on the UNCHECKED-vs-invalid distinction."""

from __future__ import annotations

import pytest

from materials_adv.evaluation.records import ValidityStatus
from materials_adv.utils.optional import has_rdkit
from materials_adv.validation.pipeline import ValidationResult, validate
from materials_adv.validation.plausibility import check_plausibility
from materials_adv.validation.representation import check_syntax, validate_representation


# --- The tri-state contract --------------------------------------------------


def test_validation_result_has_no_truth_value() -> None:
    """`if result:` would always be True and silently hide invalid candidates."""
    result = ValidationResult(representation_valid=False)
    with pytest.raises(TypeError, match="no truth value"):
        bool(result)


def test_unchecked_is_not_invalid() -> None:
    """None means 'not checked' and must never collapse to False."""
    unchecked = ValidationResult(representation_valid=None)
    invalid = ValidationResult(representation_valid=False)
    assert unchecked.status is ValidityStatus.UNCHECKED
    assert invalid.status is ValidityStatus.INVALID_REPRESENTATION
    assert unchecked.status is not invalid.status


def test_skipped_checks_are_reported() -> None:
    result = validate_representation("[*]CC[*]")
    if not has_rdkit():
        assert "rdkit_parse" in result.checks_skipped
        assert result.representation_valid is None


@pytest.mark.skipif(has_rdkit(), reason="asserts the RDKit-absent policy")
def test_without_rdkit_nothing_is_claimed_valid() -> None:
    """Strict UNCHECKED: syntax alone never yields representation_valid=True."""
    assert validate_representation("[*]CC(=O)O[*]").representation_valid is None


@pytest.mark.skipif(not has_rdkit(), reason="requires RDKit")
def test_with_rdkit_valid_string_parses() -> None:
    assert validate_representation("[*]CC(=O)O[*]").representation_valid is True


@pytest.mark.skipif(not has_rdkit(), reason="requires RDKit")
def test_with_rdkit_bad_valence_rejected() -> None:
    """Syntactically fine, chemically impossible -- only RDKit catches this."""
    assert validate_representation("C(C)(C)(C)(C)C").representation_valid is False


# --- Syntax ------------------------------------------------------------------


def test_syntax_accepts_wellformed() -> None:
    flags, reasons = check_syntax("[*]CC(=O)O[*]")
    assert all(flags.values()) and not reasons


def test_syntax_rejects_unbalanced_parens() -> None:
    flags, reasons = check_syntax("CC(C")
    assert flags["balanced_branches"] is False
    assert any("parenthes" in r for r in reasons)


def test_syntax_rejects_unpaired_ring_closure() -> None:
    flags, reasons = check_syntax("C1CC")
    assert flags["paired_ring_closures"] is False
    assert any("ring closure" in r for r in reasons)


def test_syntax_rejects_empty() -> None:
    flags, reasons = check_syntax("")
    assert flags["non_empty"] is False and reasons


def test_untokenizable_is_invalid_not_unchecked() -> None:
    """A malformed string is a real failure, distinct from 'not checked'."""
    result = validate_representation("CC{C")
    assert result.representation_valid is False
    assert result.status is ValidityStatus.INVALID_REPRESENTATION


# --- Plausibility ------------------------------------------------------------


def test_plausibility_requires_attachment_points() -> None:
    flags, reasons = check_plausibility("CCO")
    assert flags["has_attachment_points"] is False
    assert any("attachment" in r for r in reasons)


def test_plausibility_accepts_two_attachment_points() -> None:
    flags, _ = check_plausibility("[*]CC(=O)O[*]")
    assert flags["has_attachment_points"] and flags["two_attachment_points"]


def test_plausibility_flags_wrong_attachment_count() -> None:
    flags, reasons = check_plausibility("[*]CCC")
    assert flags["two_attachment_points"] is False
    assert any("found 1" in r for r in reasons)


def test_plausibility_flags_uncommon_elements() -> None:
    flags, reasons = check_plausibility("[*]C[U]C[*]")
    assert flags["common_elements_only"] is False
    assert any("uncommon element" in r for r in reasons)


def test_plausibility_flags_are_named_and_auditable() -> None:
    """Separate named flags keep the filter's selectivity reportable per-check."""
    flags, _ = check_plausibility("[*]CC[*]")
    assert set(flags) >= {
        "has_attachment_points",
        "two_attachment_points",
        "length_within_bounds",
        "has_backbone_atoms",
        "common_elements_only",
    }


# --- Combined pipeline -------------------------------------------------------


def test_pipeline_skips_plausibility_on_invalid_representation() -> None:
    result = validate("CC{C")
    assert result.representation_valid is False
    assert "plausibility" in result.checks_skipped


def test_pipeline_small_molecule_is_implausible_when_representation_ok() -> None:
    result = validate("CCO")
    if result.representation_valid is not False:
        assert result.plausibility_flags["has_attachment_points"] is False


def test_plausible_property_requires_flags_present() -> None:
    """Empty flags must not read as 'plausible' by vacuous truth."""
    assert ValidationResult(representation_valid=True, plausibility_flags={}).plausible is False
