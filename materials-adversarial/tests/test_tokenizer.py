"""Tokenizer tests, including the regressions that pin the regex ordering."""

from __future__ import annotations

import pytest

from materials_adv.data.tokenizer import (
    PAD_TOKEN,
    PSmilesTokenizer,
    TokenizationError,
    UNK_TOKEN,
    Vocabulary,
    detokenize,
    tokenize,
    unknown_character_histogram,
)

VALID_CORPUS = [
    "[*]CC(=O)O[*]",
    "*CC(c1ccccc1)*",
    "CC(Cl)Br",
    "N[C@@H](C)C(=O)O",
    "C1CC%10CC%10C1",
    "[nH]1cccc1",
    "C%(1234)CC%(1234)",
    "[13CH4]",
    "O=C(N)[Si](C)(C)C",
    "C1CC1.C1CC1",
    "cc[se]cc",
    "C/C=C/C",
    "CC(=O)[O-]",
    "[*]Oc1ccc(cc1)C(C)(C)c1ccc(O[*])cc1",
]


@pytest.mark.parametrize("smiles", VALID_CORPUS)
def test_round_trip_is_exact(smiles: str) -> None:
    assert detokenize(tokenize(smiles)) == smiles


# --- Regression tests for the load-bearing alternation order -----------------
# If the single-character class is moved before `Br|Cl`, 'Cl' matches 'C' and
# leaves a stray 'l', silently turning chlorine into carbon.


def test_chlorine_is_one_token_not_carbon_plus_l() -> None:
    tokens = tokenize("CC(Cl)Br")
    assert "Cl" in tokens
    assert "l" not in tokens
    assert tokens == ["C", "C", "(", "Cl", ")", "Br"]


def test_bromine_is_one_token_not_boron_plus_r() -> None:
    tokens = tokenize("BrCCBr")
    assert tokens == ["Br", "C", "C", "Br"]
    assert "r" not in tokens


def test_bracket_atom_does_not_fragment() -> None:
    assert tokenize("[nH]1cccc1")[0] == "[nH]"
    assert tokenize("N[C@@H](C)C")[1] == "[C@@H]"


def test_two_digit_ring_closure_is_one_token() -> None:
    tokens = tokenize("C1CC%10CC%10C1")
    assert tokens.count("%10") == 2
    assert "%" not in tokens


def test_extended_ring_closure_is_one_token() -> None:
    assert "%(1234)" in tokenize("C%(1234)CC%(1234)")


def test_case_is_preserved_aromatic_vs_aliphatic() -> None:
    """'C' is aliphatic carbon, 'c' is aromatic. Case folding would corrupt chemistry."""
    assert tokenize("Cc") == ["C", "c"]


# --- Unknown-character handling ---------------------------------------------


def test_unknown_raises_by_default() -> None:
    with pytest.raises(TokenizationError) as exc:
        tokenize("CC{C")
    assert exc.value.char == "{"
    assert exc.value.position == 2


def test_unknown_char_mode_preserves_round_trip() -> None:
    text = "CC{C"
    assert detokenize(tokenize(text, on_unknown="char")) == text


def test_unknown_unk_mode_breaks_round_trip_as_documented() -> None:
    tokens = tokenize("CC{C", on_unknown="unk")
    assert UNK_TOKEN in tokens
    assert detokenize(tokens) != "CC{C"


def test_finditer_style_silent_skip_does_not_happen() -> None:
    """An unrecognised character must never be silently dropped."""
    tokens = tokenize("CC{C", on_unknown="char")
    assert "{" in tokens


def test_unknown_histogram_is_empty_on_valid_corpus() -> None:
    assert unknown_character_histogram(VALID_CORPUS) == {}


def test_unknown_histogram_counts_offenders() -> None:
    hist = unknown_character_histogram(["CC{C", "C}C{"])
    assert hist["{"] == 2
    assert hist["}"] == 1


# --- Vocabulary --------------------------------------------------------------


def test_pad_is_id_zero() -> None:
    vocab = Vocabulary.build(VALID_CORPUS)
    assert vocab.itos[0] == PAD_TOKEN
    assert vocab.pad_id == 0


def test_vocab_construction_is_deterministic_under_row_reordering() -> None:
    """The lexicographic tiebreak must make ordering independent of corpus order."""
    a = Vocabulary.build(VALID_CORPUS)
    b = Vocabulary.build(list(reversed(VALID_CORPUS)))
    assert a.itos == b.itos


def test_encode_decode_round_trip_in_vocab() -> None:
    vocab = Vocabulary.build(VALID_CORPUS)
    tok = PSmilesTokenizer(vocab)
    for smiles in VALID_CORPUS:
        assert tok.decode(tok.encode(smiles)) == smiles


def test_encode_decode_with_special_tokens_strips_them() -> None:
    vocab = Vocabulary.build(VALID_CORPUS)
    tok = PSmilesTokenizer(vocab)
    smiles = "CC(Cl)Br"
    ids = tok.encode(smiles, add_special_tokens=True)
    assert tok.decode(ids, skip_special_tokens=True) == smiles


def test_oov_breaks_encode_decode_round_trip_as_documented() -> None:
    vocab = Vocabulary.build(["CCC"])
    tok = PSmilesTokenizer(vocab)
    assert tok.decode(tok.encode("CCN")) != "CCN"


def test_encode_requires_vocabulary() -> None:
    with pytest.raises(ValueError, match="no vocabulary"):
        PSmilesTokenizer().encode("CCC")


def test_tokenize_works_without_vocabulary() -> None:
    assert PSmilesTokenizer().tokenize("CC(Cl)Br") == ["C", "C", "(", "Cl", ")", "Br"]


def test_vocab_save_load_round_trip(tmp_path) -> None:
    vocab = Vocabulary.build(VALID_CORPUS)
    path = vocab.save(tmp_path / "vocab.json")
    assert Vocabulary.load(path).itos == vocab.itos
