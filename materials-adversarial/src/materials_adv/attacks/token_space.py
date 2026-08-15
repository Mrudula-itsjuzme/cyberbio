"""Structural analysis of a PSMILES token list.

Attacks operate here, on `list[str]`, never on raw strings.

The central primitive is `editable_positions`. Some edits are *guaranteed* to
produce an invalid representation:

  - deleting one parenthesis of a matched pair
  - deleting one digit of a matched ring-closure pair
  - removing a polymer attachment point ('*' / '[*]'), after which the string is
    no longer a repeat unit at all

An "attack" made of such edits would measure the validity filter's ability to
reject broken strings, not the model's robustness. Those positions are therefore
protected by default. Protection is a config flag rather than a hardcoded rule,
so the effect of relaxing it remains measurable rather than assumed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum

RING_CLOSURE_RE = re.compile(r"^(?:\d|%\d{2}|%\(\d{3,5}\))$")
BOND_TOKENS = frozenset({"=", "#", "$", ":", "/", "\\", "-", "~", "+", "@", "?", ">"})
DISCONNECT_TOKENS = frozenset({"."})


class TokenRole(str, Enum):
    ALIPHATIC_ATOM = "aliphatic_atom"  # C, N, O, Cl
    AROMATIC_ATOM = "aromatic_atom"    # c, n, o
    BRACKET_ATOM = "bracket_atom"    # [nH], [C@@H], [Si], [O-]
    ATTACHMENT = "attachment"        # polymer connection point: * or [*]
    RING_CLOSURE = "ring_closure"    # 1, %10, %(1234)
    BRANCH_OPEN = "branch_open"      # (
    BRANCH_CLOSE = "branch_close"    # )
    BOND = "bond"                    # = # $ : / \ - ~
    DISCONNECT = "disconnect"        # .
    OTHER = "other"


def is_attachment(token: str) -> bool:
    """True for a polymer attachment/connection point.

    Covers bare '*' and bracket forms like '[*]' and '[*:1]'. Note that RDKit
    parses these as dummy atoms without validating that the connection topology
    forms a sensible repeat unit -- see validation/plausibility.py.
    """
    if token == "*":
        return True
    return token.startswith("[") and "*" in token


def classify_token(token: str) -> TokenRole:
    if is_attachment(token):
        return TokenRole.ATTACHMENT
    if token.startswith("["):
        return TokenRole.BRACKET_ATOM
    if token == "(":
        return TokenRole.BRANCH_OPEN
    if token == ")":
        return TokenRole.BRANCH_CLOSE
    if RING_CLOSURE_RE.match(token):
        return TokenRole.RING_CLOSURE
    if token in DISCONNECT_TOKENS:
        return TokenRole.DISCONNECT
    if token in BOND_TOKENS:
        return TokenRole.BOND
    if token.isalpha():
        if token.isupper() or token == "Cl" or token == "Br": # Basic handling for 2-letter elements, but islower captures aromatic usually.
            # wait, Cl and Br are aliphatic, they start with uppercase
            return TokenRole.ALIPHATIC_ATOM if token[0].isupper() else TokenRole.AROMATIC_ATOM
        return TokenRole.AROMATIC_ATOM
    return TokenRole.OTHER


def classify(tokens: list[str]) -> list[TokenRole]:
    return [classify_token(t) for t in tokens]


def ring_closure_pairs(tokens: list[str]) -> dict[str, list[int]]:
    """Map each ring-closure label to the positions where it appears.

    A well-formed structure has each label appearing an even number of times
    (SMILES permits digit reuse after a pair closes, so this is a grouping, not
    a strict pairing).
    """
    positions: dict[str, list[int]] = defaultdict(list)
    for i, tok in enumerate(tokens):
        if classify_token(tok) is TokenRole.RING_CLOSURE:
            positions[tok].append(i)
    return dict(positions)


def editable_positions(
    tokens: list[str],
    *,
    protect_attachments: bool = True,
    protect_ring_closures: bool = True,
    protect_branches: bool = True,
) -> tuple[int, ...]:
    """Indices an attack may safely edit.

    Protecting a position is not a claim that editing it is chemically wrong --
    only that editing it makes invalidity a certainty, which would confound the
    measurement of prediction drift.
    """
    protected: set[int] = set()

    for i, tok in enumerate(tokens):
        role = classify_token(tok)
        if protect_attachments and role is TokenRole.ATTACHMENT:
            protected.add(i)
        if protect_branches and role in (TokenRole.BRANCH_OPEN, TokenRole.BRANCH_CLOSE):
            protected.add(i)
        if protect_ring_closures and role is TokenRole.RING_CLOSURE:
            protected.add(i)
        if role is TokenRole.DISCONNECT:
            protected.add(i)

    return tuple(i for i in range(len(tokens)) if i not in protected)


def count_changes(original: list[str], modified: list[str]) -> int:
    """Token-level edit distance (Levenshtein over tokens).

    Used for `number_of_changes`. Computed rather than trusted from the attack's
    own bookkeeping, so a buggy attack cannot under-report its perturbation size.
    """
    n, m = len(original), len(modified)
    if n == 0:
        return m
    if m == 0:
        return n

    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if original[i - 1] == modified[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[m]
