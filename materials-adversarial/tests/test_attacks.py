"""Attack mechanics tests, on synthetic token lists (no data, no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from materials_adv.attacks.base import AttackOutcome
from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.generator import AttackGenerator, ConstantPredictor
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.registry import available_attacks, build_attack
from materials_adv.attacks.reordering import ReorderingAttack
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
        ("C", TokenRole.ATOM),
        ("[nH]", TokenRole.BRACKET_ATOM),
        ("(", TokenRole.BRANCH_OPEN),
        (")", TokenRole.BRANCH_CLOSE),
        ("1", TokenRole.RING_CLOSURE),
        ("%10", TokenRole.RING_CLOSURE),
        ("=", TokenRole.BOND),
    ],
)
def test_classify_token(token: str, expected: TokenRole) -> None:
    assert classify_token(token) is expected


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

ATTACKS = [
    lambda r: DeletionAttack(r),
    lambda r: InsertionAttack(r),
    lambda r: ReorderingAttack(r),
    lambda r: SubstitutionAttack(r, allowed_tokens=["C", "N", "O", "S"]),
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
    for o in InsertionAttack(rng(), n_edits=1).generate(list(TOKENS), n_variants=3):
        assert len(o.adversarial_tokens) == len(TOKENS) + 1


def test_reordering_preserves_token_multiset() -> None:
    """A local rearrangement must not create or destroy tokens."""
    for o in ReorderingAttack(rng(), window=3).generate(list(TOKENS), n_variants=5):
        assert sorted(o.adversarial_tokens) == sorted(o.original_tokens)


def test_reordering_is_local_within_window() -> None:
    window = 2
    for o in ReorderingAttack(rng(5), window=window).generate(list(TOKENS), n_variants=5):
        moved = [
            i
            for i, (a, b) in enumerate(zip(o.original_tokens, o.adversarial_tokens))
            if a != b
        ]
        if moved:
            assert max(moved) - min(moved) <= window


def test_substitution_preserves_length() -> None:
    attack = SubstitutionAttack(rng(), allowed_tokens=["C", "N", "O", "S"])
    for o in attack.generate(list(TOKENS), n_variants=3):
        assert len(o.adversarial_tokens) == len(TOKENS)


def test_substitution_role_preserving_swaps_atom_for_atom() -> None:
    attack = SubstitutionAttack(
        rng(1), allowed_tokens=["C", "N", "O", "=", "("], role_preserving=True
    )
    for o in attack.generate(list(TOKENS), n_variants=10):
        for pos in o.edit_positions:
            assert classify_token(o.adversarial_tokens[pos]) is classify_token(
                o.original_tokens[pos]
            )


def test_substitution_without_pool_raises_pending() -> None:
    """The replacement pool must be data-derived, never hardcoded."""
    with pytest.raises(PendingImplementation) as exc:
        SubstitutionAttack(rng())
    assert exc.value.blocked_on == "dataset"


def test_reordering_rejects_degenerate_window() -> None:
    with pytest.raises(ValueError, match="window must be >= 2"):
        ReorderingAttack(rng(), window=1)


# --- Registry ----------------------------------------------------------------


def test_all_four_attacks_registered() -> None:
    assert set(available_attacks()) >= {
        "substitution",
        "insertion",
        "deletion",
        "reordering",
    }


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
        attacks=[DeletionAttack(rng()), InsertionAttack(rng())],
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
