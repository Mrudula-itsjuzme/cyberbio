"""Attack mechanics tests, on synthetic token lists (no data, no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from materials_adv.attacks.base import AttackOutcome
from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.generator import AttackGenerator, ConstantPredictor
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.registry import available_attacks, build_attack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.attacks.token_space import (
    TokenRole,
    classify_token,
    count_changes,
    editable_positions,
    is_attachment,
)
from materials_adv.data.tokenizer import tokenize
from materials_adv.utils.pending import PendingImplementation

POLYMER = "[*]CC(=O)O[*]"
TOKENS = tokenize(POLYMER)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- Token space -------------------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("[*]", TokenRole.ATTACHMENT),
        ("*", TokenRole.ATTACHMENT),
        # Phase 1D split the former single ATOM role into aliphatic vs aromatic,
        # because role-preserving substitution was swapping across that boundary
        # and producing invalid strings.
        ("C", TokenRole.ALIPHATIC_ATOM),
        ("N", TokenRole.ALIPHATIC_ATOM),
        ("Cl", TokenRole.ALIPHATIC_ATOM),
        ("Br", TokenRole.ALIPHATIC_ATOM),
        ("c", TokenRole.AROMATIC_ATOM),
        ("n", TokenRole.AROMATIC_ATOM),
        ("[nH]", TokenRole.BRACKET_ATOM),
        ("(", TokenRole.BRANCH_OPEN),
        (")", TokenRole.BRANCH_CLOSE),
        ("1", TokenRole.RING_CLOSURE),
        ("%10", TokenRole.RING_CLOSURE),
        ("=", TokenRole.BOND),
        ("/", TokenRole.BOND),
        # Phase 1D separated '.' from BOND: substitution was admitting '=' -> '.',
        # which disconnects the molecule rather than perturbing it.
        (".", TokenRole.DISCONNECT),
    ],
)
def test_classify_token(token: str, expected: TokenRole) -> None:
    assert classify_token(token) is expected


def test_aliphatic_and_aromatic_are_distinct_roles() -> None:
    """Regression for the Phase 1D taxonomy fix.

    'C' and 'c' are different chemistry. Collapsing them into one role let
    role-preserving substitution swap aliphatic for aromatic carbon.
    """
    assert classify_token("C") is not classify_token("c")


def test_disconnect_is_not_a_bond() -> None:
    """Regression: '.' disconnects the molecule; it must not be substitutable as a bond."""
    assert classify_token(".") is not classify_token("=")
    assert classify_token(".") is TokenRole.DISCONNECT


def test_disconnect_is_never_editable() -> None:
    """A '.' token is protected unconditionally, independent of the protection flags."""
    tokens = ["C", ".", "C"]
    editable = editable_positions(
        tokens,
        protect_attachments=False,
        protect_ring_closures=False,
        protect_branches=False,
    )
    assert 1 not in editable


def test_attachment_detection() -> None:
    assert is_attachment("*") and is_attachment("[*]") and is_attachment("[*:1]")
    assert not is_attachment("C") and not is_attachment("[nH]")


def test_protected_positions_excluded() -> None:
    editable = editable_positions(TOKENS)
    for i in editable:
        assert classify_token(TOKENS[i]) is not TokenRole.ATTACHMENT
        assert TOKENS[i] not in ("(", ")")


def test_protection_can_be_relaxed() -> None:
    """Protection is a measurable flag, not a hardcoded rule."""
    strict = editable_positions(TOKENS)
    relaxed = editable_positions(
        TOKENS, protect_attachments=False, protect_ring_closures=False, protect_branches=False
    )
    assert len(relaxed) > len(strict)
    assert len(relaxed) == len(TOKENS)


def test_count_changes_is_token_level() -> None:
    assert count_changes(["C", "C"], ["C", "C"]) == 0
    assert count_changes(["C", "C"], ["C", "N"]) == 1
    assert count_changes(["C", "C"], ["C"]) == 1
    assert count_changes(["C"], ["C", "N", "O"]) == 2


# --- Shared attack contracts -------------------------------------------------

# InsertionAttack now takes allowed_tokens as a required argument (it moved from
# "sample the sequence's own tokens" to an explicit injected pool, matching
# substitution). The pool below is synthetic and used only to exercise mechanics.
POOL = ["C", "N", "O", "S", "c", "="]

ATTACKS = [
    lambda r: DeletionAttack(r),
    lambda r: InsertionAttack(r, allowed_tokens=POOL),
    lambda r: RearrangementAttack(r, window_size=3),
    lambda r: SubstitutionAttack(r, allowed_tokens=POOL),
]


@pytest.mark.parametrize("factory", ATTACKS)
def test_attack_does_not_mutate_input(factory) -> None:
    tokens = list(TOKENS)
    factory(rng()).generate(tokens, n_variants=3)
    assert tokens == list(TOKENS)


@pytest.mark.parametrize("factory", ATTACKS)
def test_attack_is_deterministic_given_seed(factory) -> None:
    a = factory(rng(42)).generate(list(TOKENS), n_variants=3)
    b = factory(rng(42)).generate(list(TOKENS), n_variants=3)
    assert [o.adversarial_tokens for o in a] == [o.adversarial_tokens for o in b]


@pytest.mark.parametrize("factory", ATTACKS)
def test_attack_never_edits_protected_positions(factory) -> None:
    """Attachment points must survive: without them it is not a repeat unit."""
    for outcome in factory(rng(7)).generate(list(TOKENS), n_variants=10):
        n_before = sum(1 for t in outcome.original_tokens if is_attachment(t))
        n_after = sum(1 for t in outcome.adversarial_tokens if is_attachment(t))
        assert n_after == n_before


@pytest.mark.parametrize("factory", ATTACKS)
def test_number_of_changes_matches_actual_diff(factory) -> None:
    for outcome in factory(rng(3)).generate(list(TOKENS), n_variants=5):
        expected = count_changes(
            list(outcome.original_tokens), list(outcome.adversarial_tokens)
        )
        assert outcome.number_of_changes == expected
        assert outcome.number_of_changes > 0


# --- Per-attack specifics ----------------------------------------------------


def test_deletion_shortens_sequence() -> None:
    for o in DeletionAttack(rng(), n_edits=1).generate(list(TOKENS), n_variants=3):
        assert len(o.adversarial_tokens) == len(TOKENS) - 1


def test_insertion_lengthens_sequence() -> None:
    for o in InsertionAttack(rng(), allowed_tokens=POOL, n_edits=1).generate(
        list(TOKENS), n_variants=3
    ):
        assert len(o.adversarial_tokens) == len(TOKENS) + 1


def test_rearrangement_preserves_token_multiset() -> None:
    """A local rearrangement must not create or destroy tokens."""
    for o in RearrangementAttack(rng(), window_size=3).generate(list(TOKENS), n_variants=5):
        assert sorted(o.adversarial_tokens) == sorted(o.original_tokens)


def test_rearrangement_is_local_within_window() -> None:
    window = 3
    for o in RearrangementAttack(rng(5), window_size=window).generate(
        list(TOKENS), n_variants=5
    ):
        moved = [
            i
            for i, (a, b) in enumerate(zip(o.original_tokens, o.adversarial_tokens))
            if a != b
        ]
        if moved:
            assert max(moved) - min(moved) <= window


def test_substitution_preserves_length() -> None:
    attack = SubstitutionAttack(rng(), allowed_tokens=POOL)
    for o in attack.generate(list(TOKENS), n_variants=3):
        assert len(o.adversarial_tokens) == len(TOKENS)


def test_substitution_role_preserving_swaps_within_role() -> None:
    """Role preservation now respects the finer Phase 1D taxonomy.

    A pool spanning aliphatic, aromatic, bond and branch roles must never yield
    a cross-role swap -- that was the Phase 1D bug ('=' -> '.').
    """
    attack = SubstitutionAttack(
        rng(1), allowed_tokens=["C", "N", "O", "c", "n", "=", "(", "."], role_preserving=True
    )
    for o in attack.generate(list(TOKENS), n_variants=10):
        for pos in o.edit_positions:
            assert classify_token(o.adversarial_tokens[pos]) is classify_token(
                o.original_tokens[pos]
            )


def test_substitution_never_introduces_disconnect() -> None:
    """Regression for the Phase 1D smoke-test finding: '=' must not become '.'."""
    attack = SubstitutionAttack(
        rng(2), allowed_tokens=["C", "N", "=", "."], role_preserving=True
    )
    for o in attack.generate(list(TOKENS), n_variants=20):
        n_before = sum(1 for t in o.original_tokens if t == ".")
        n_after = sum(1 for t in o.adversarial_tokens if t == ".")
        assert n_after == n_before


def test_substitution_requires_explicit_pool() -> None:
    """The replacement pool is injected, never hardcoded.

    Substitution retains its explicit PendingImplementation guard, so omitting
    the pool fails with an actionable message. Insertion took the other route
    (a required positional argument) -- both refuse to invent a chemical prior,
    noted as an inconsistency in Problems #13.
    """
    with pytest.raises(PendingImplementation) as exc:
        SubstitutionAttack(rng())
    assert exc.value.blocked_on == "dataset"


def test_rearrangement_degenerate_window_yields_no_candidates() -> None:
    """window_size=1 cannot contain a swap.

    The scaffold version raised ValueError here. The Phase 1E rewrite does not
    validate the argument; it degrades to producing zero candidates instead.
    That is safe (no silent no-op candidates enter the record), so this test
    pins the ACTUAL behaviour rather than asserting a guard that no longer
    exists. Adding explicit validation is noted in Problems #13.
    """
    assert RearrangementAttack(rng(), window_size=1).generate(list(TOKENS), n_variants=5) == []


# --- Registry ----------------------------------------------------------------


def test_all_four_attacks_registered() -> None:
    """Regression: the Phase 1D/1E rewrites dropped @register_attack from three
    of the four attacks, leaving only 'substitution' registered and
    build_attack('deletion') raising KeyError. Experiments construct attacks
    directly so results were unaffected, but the pluggability contract was
    broken. Decorators restored in Phase 2C.
    """
    assert set(available_attacks()) >= {
        "substitution",
        "insertion",
        "deletion",
        "rearrangement",
    }


def test_every_registered_attack_is_buildable() -> None:
    required_kwargs = {
        "insertion": {"allowed_tokens": POOL},
        "substitution": {"allowed_tokens": POOL},
    }
    for name in ("substitution", "insertion", "deletion", "rearrangement"):
        attack = build_attack(name, rng(), **required_kwargs.get(name, {}))
        assert attack.metadata()["attack_type"] == name


def test_build_attack_by_name() -> None:
    attack = build_attack("deletion", rng())
    assert isinstance(attack, DeletionAttack)
    assert attack.metadata()["attack_type"] == "deletion"


# --- Outcome -----------------------------------------------------------------


def test_noop_detection() -> None:
    o = AttackOutcome(
        original_tokens=("C", "C"), adversarial_tokens=("C", "C"), attack_type="t"
    )
    assert o.is_noop and o.number_of_changes == 0


# --- End-to-end pipeline without torch ---------------------------------------


def test_generator_end_to_end_with_constant_predictor() -> None:
    gen = AttackGenerator(
        attacks=[DeletionAttack(rng()), InsertionAttack(rng(), allowed_tokens=POOL)],
        predictor=ConstantPredictor(300.0),
        seed=1,
    )
    records = gen.run([("s1", POLYMER)], n_variants=2)
    assert records
    for r in records:
        assert r.sample_id == "s1"
        assert r.original_prediction == 300.0
        assert r.prediction_drift == 0.0  # constant predictor => zero drift
        assert r.attack_id.startswith("s1:")


def test_generator_without_predictor_leaves_predictions_none() -> None:
    gen = AttackGenerator(attacks=[DeletionAttack(rng())], seed=1)
    for r in gen.run([("s1", POLYMER)]):
        assert r.original_prediction is None
        assert r.prediction_drift is None


def test_generator_requires_at_least_one_attack() -> None:
    with pytest.raises(ValueError, match="at least one attack"):
        AttackGenerator(attacks=[])
