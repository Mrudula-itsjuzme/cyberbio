"""Representation-level validity: is this a well-formed string?

Two distinct layers, deliberately not merged:

  SYNTACTIC   -- bracket/paren/ring balance and tokenizability. Always available,
                 needs no dependencies. Proves the string is well-formed, and
                 NOTHING about chemistry.

  CHEMICAL    -- RDKit parse and sanitization. Proves the valences and ring
                 closures describe a parseable molecule.

Per the project's strict-UNCHECKED policy, passing the syntactic layer does NOT
set `representation_valid = True`. Only a successful RDKit parse can do that.
Without RDKit every candidate is UNCHECKED, and no Phase 1 result may claim
chemical validity.
"""

from __future__ import annotations

from ..data.tokenizer import TokenizationError, tokenize
from ..utils.optional import has_rdkit
from .pipeline import ValidationResult


def check_syntax(psmiles: str) -> tuple[dict[str, bool], list[str]]:
    """Cheap structural checks. Returns (flags, failure reasons)."""
    flags: dict[str, bool] = {}
    reasons: list[str] = []

    if not psmiles or not psmiles.strip():
        return {"non_empty": False}, ["empty string"]
    flags["non_empty"] = True

    try:
        tokens = tokenize(psmiles)
        flags["tokenizable"] = True
    except TokenizationError as exc:
        flags["tokenizable"] = False
        reasons.append(f"untokenizable: {exc.char!r} at position {exc.position}")
        return flags, reasons

    depth = 0
    balanced = True
    for tok in tokens:
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth < 0:
                balanced = False
                break
    balanced = balanced and depth == 0
    flags["balanced_branches"] = balanced
    if not balanced:
        reasons.append("unbalanced branch parentheses")

    from ..attacks.token_space import ring_closure_pairs

    unpaired = [label for label, pos in ring_closure_pairs(tokens).items() if len(pos) % 2 != 0]
    flags["paired_ring_closures"] = not unpaired
    if unpaired:
        reasons.append(f"unpaired ring closure(s): {sorted(unpaired)}")

    return flags, reasons


def check_rdkit(psmiles: str) -> tuple[bool | None, list[str]]:
    """RDKit parse. Returns (valid, reasons); valid is None when RDKit is absent.

    Star/attachment atoms ('*', '[*]') are parsed by RDKit as dummy atoms. RDKit
    validates the resulting graph but has NO notion of whether the connection
    points form a sensible polymer repeat unit -- that is a plausibility concern,
    handled separately in validation/plausibility.py.
    """
    if not has_rdkit():
        return None, []

    from rdkit import Chem, RDLogger  # noqa: PLC0415 -- lazy by design

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(psmiles, sanitize=True)
    except Exception as exc:  # RDKit raises a variety of C++-backed exceptions
        return False, [f"rdkit parse raised: {type(exc).__name__}: {exc}"]

    if mol is None:
        return False, ["rdkit could not parse the string"]
    return True, []


def validate_representation(psmiles: str) -> ValidationResult:
    """Full representation check, honest about what was and was not verified."""
    flags, reasons = check_syntax(psmiles)
    checks_run = ["syntax"]
    checks_skipped: list[str] = []

    syntax_ok = all(flags.values())

    if not syntax_ok:
        # A string that will not tokenize cannot be handed to RDKit meaningfully.
        checks_skipped.append("rdkit_parse")
        return ValidationResult(
            representation_valid=False,
            plausibility_flags=flags,
            rejection_reasons=tuple(reasons),
            checks_run=tuple(checks_run),
            checks_skipped=tuple(checks_skipped),
        )

    rdkit_valid, rdkit_reasons = check_rdkit(psmiles)
    if rdkit_valid is None:
        checks_skipped.append("rdkit_parse")
        # Syntactically fine, chemically unverified => UNCHECKED, never True.
        return ValidationResult(
            representation_valid=None,
            plausibility_flags=flags,
            rejection_reasons=(),
            checks_run=tuple(checks_run),
            checks_skipped=tuple(checks_skipped),
        )

    checks_run.append("rdkit_parse")
    return ValidationResult(
        representation_valid=rdkit_valid,
        plausibility_flags=flags,
        rejection_reasons=tuple(reasons + rdkit_reasons),
        checks_run=tuple(checks_run),
        checks_skipped=tuple(checks_skipped),
    )
