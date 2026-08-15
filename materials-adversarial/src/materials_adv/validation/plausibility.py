"""Polymer plausibility heuristics.

READ THIS BEFORE CITING ANY RESULT FROM THIS MODULE.

These are cheap, explicit, necessary-but-not-sufficient checks. Passing them does
NOT establish that a string is a real, synthesizable polymer with a meaningful
Tg. No automated check available here can establish that.

What these checks can do is REJECT obviously broken candidates (no attachment
points, exotic elements, absurd size). What they cannot do is ACCEPT anything as
scientifically plausible. Any writeup must describe survivors as "not rejected by
heuristic filters", never as "chemically plausible polymers".

Each check is a separate named flag so rejection reasons stay auditable and the
filter's own selectivity can be reported per-check.
"""

from __future__ import annotations

from ..attacks.token_space import TokenRole, classify_token, is_attachment
from ..data.tokenizer import TokenizationError, tokenize

# Elements common in organic polymer repeat units. Deliberately conservative: a
# candidate containing something outside this set is FLAGGED for review, not
# proven wrong. Widen this from the actual dataset once it is audited.
COMMON_POLYMER_ELEMENTS: frozenset[str] = frozenset(
    {"C", "c", "N", "n", "O", "o", "S", "s", "P", "p", "F", "Cl", "Br", "I", "Si", "B", "H"}
)

MIN_TOKENS = 3
MAX_TOKENS = 400


def _element_symbols(tokens: list[str]) -> set[str]:
    """Extract element symbols, including from inside bracket atoms."""
    symbols: set[str] = set()
    for t in tokens:
        role = classify_token(t)
        if role in (TokenRole.ALIPHATIC_ATOM, TokenRole.AROMATIC_ATOM):
            # C, c, N, n, Cl -> C, N, Cl (capitalized standard symbol)
            symbols.add(t.capitalize())
        elif role is TokenRole.BRACKET_ATOM:
            inner = t.strip("[]")
            # Strip isotope digits, charge, chirality, H-count annotations.
            i = 0
            while i < len(inner) and inner[i].isdigit():
                i += 1
            if i < len(inner):
                if i + 1 < len(inner) and inner[i : i + 2] in {"Cl", "Br", "Si", "Se", "se"}:
                    symbols.add(inner[i : i + 2])
                elif inner[i].isalpha():
                    symbols.add(inner[i])
    return symbols


def check_plausibility(psmiles: str) -> tuple[dict[str, bool], list[str]]:
    """Heuristic polymer checks. Returns (flags, reasons for any failures)."""
    flags: dict[str, bool] = {}
    reasons: list[str] = []

    try:
        tokens = tokenize(psmiles)
    except TokenizationError:
        return {"tokenizable_for_plausibility": False}, ["untokenizable"]

    # 1. Attachment points. A repeat unit needs exactly two connection points;
    #    without them the string is a small molecule, not a polymer unit.
    n_attach = sum(1 for t in tokens if is_attachment(t))
    flags["has_attachment_points"] = n_attach > 0
    flags["two_attachment_points"] = n_attach == 2
    if n_attach == 0:
        reasons.append("no polymer attachment point ('*' or '[*]') present")
    elif n_attach != 2:
        reasons.append(f"expected 2 attachment points for a linear repeat unit, found {n_attach}")

    # 2. Size bounds.
    flags["length_within_bounds"] = MIN_TOKENS <= len(tokens) <= MAX_TOKENS
    if not flags["length_within_bounds"]:
        reasons.append(f"token count {len(tokens)} outside [{MIN_TOKENS}, {MAX_TOKENS}]")

    # 3. Backbone atoms present.
    n_atoms = sum(
        1
        for t in tokens
        if classify_token(t) in (TokenRole.ALIPHATIC_ATOM, TokenRole.AROMATIC_ATOM, TokenRole.BRACKET_ATOM)
    )
    flags["has_backbone_atoms"] = n_atoms >= 1
    if n_atoms < 1:
        reasons.append("no non-attachment atoms present")

    # 4. Element whitelist -- flags the unusual, does not prove the usual correct.
    unexpected = _element_symbols(tokens) - COMMON_POLYMER_ELEMENTS
    flags["common_elements_only"] = not unexpected
    if unexpected:
        reasons.append(f"uncommon element(s) for an organic polymer: {sorted(unexpected)}")

    return flags, reasons
