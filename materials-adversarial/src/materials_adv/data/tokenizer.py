"""PSMILES chemical tokenizer.

Tokenization operates on *chemically meaningful* units, not characters. A
character-level split would tear 'Cl' into 'C' + 'l' and '[C@@H]' into fragments,
so any downstream "attack" on such a split would be measuring a string-handling
bug rather than model robustness.

The tokenization RULES are data-independent (fixed SMILES grammar) and therefore
fully implementable and testable now. Only the VOCABULARY is data-derived, so
`build_vocab` is separate from `tokenize` and `encode`/`decode` require a vocab.

Round-trip guarantee
--------------------
    detokenize(tokenize(x)) == x

holds EXACTLY whenever tokenization completes without unknown characters, because
tokenize produces a non-overlapping partition of the input and detokenize is a
plain join. No normalization and no case folding are applied (case is meaningful
in SMILES: 'C' is aliphatic carbon, 'c' is aromatic).

What breaks the round trip -- all documented in docs/PROJECT_WORKSPACE.md S4:
  1. encode -> decode instead of tokenize -> detokenize. Out-of-vocabulary tokens
     map to <unk> and the original text is unrecoverable.
  2. Upstream canonicalization (e.g. RDKit) changing the string before tokenizing.
  3. on_unknown='unk', which replaces unknown characters with a placeholder.
  4. Special tokens: identity holds only after they are stripped.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------
# Token grammar
#
# ORDER IS LOAD-BEARING. Alternation is first-match, so:
#   - bracket atoms must come first, or '[nH]' fragments into '[' + 'n' + 'H]'
#   - 'Br|Cl' must precede the single-character class, or 'Cl' matches 'C' and
#     leaves a stray 'l', silently turning chlorine into carbon.
# Both hazards are pinned by regression tests in tests/test_tokenizer.py.
# --------------------------------------------------------------------------
TOKEN_PATTERN = re.compile(
    r"(\[[^\]]*\])"            # bracket atom: [nH], [C@@H], [13CH4], [*], [Si]
    r"|(%\(\d{3,5}\))"         # extended ring closure: %(1234)
    r"|(%\d{2})"               # two-digit ring closure: %10
    r"|(Br|Cl)"                # two-letter organic-subset elements
    r"|([BCNOSPFIbcnosp*])"    # single-char organic subset + attachment point
    r"|([0-9])"                # ring closure digit
    r"|([()])"                 # branch open/close
    r"|([=#$:/\\.\-+~@?>])"    # bonds and misc structural symbols
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# <pad> is pinned to id 0 so padding is index 0 regardless of corpus.
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)

OnUnknown = Literal["raise", "unk", "char"]


class TokenizationError(ValueError):
    """Input contains a character the SMILES grammar does not recognise."""

    def __init__(self, text: str, position: int, char: str) -> None:
        self.text = text
        self.position = position
        self.char = char
        super().__init__(
            f"Unrecognised character {char!r} at position {position} in {text!r}\n"
            f"  {text}\n"
            f"  {' ' * position}^"
        )


def tokenize(text: str, *, on_unknown: OnUnknown = "raise") -> list[str]:
    """Split a PSMILES string into chemical tokens.

    Uses anchored `match` at an explicit cursor rather than `finditer`, because
    `finditer` silently SKIPS characters it cannot match -- which would drop
    unrecognised chemistry without a trace.

    on_unknown:
      'raise' (default) -- fail loudly on an unrecognised character.
      'unk'             -- emit UNK_TOKEN. Breaks the round-trip guarantee.
      'char'            -- emit the raw character as its own token. Preserves the
                           round trip and is used when building a vocabulary, so
                           unexpected grammar surfaces as data rather than a crash.
    """
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = TOKEN_PATTERN.match(text, i)
        if m is None:
            if on_unknown == "raise":
                raise TokenizationError(text, i, text[i])
            tokens.append(UNK_TOKEN if on_unknown == "unk" else text[i])
            i += 1
            continue
        tokens.append(m.group(0))
        i = m.end()
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into a string. Inverse of `tokenize` (see module docstring)."""
    return "".join(tokens)


def unknown_character_histogram(texts: list[str]) -> Counter[str]:
    """Count characters the grammar does not recognise, across a corpus.

    Diagnostic for the dataset audit: unexpected grammar becomes a reportable
    number instead of an exception or, worse, a silent drop.
    """
    hist: Counter[str] = Counter()
    for text in texts:
        i = 0
        while i < len(text):
            m = TOKEN_PATTERN.match(text, i)
            if m is None:
                hist[text[i]] += 1
                i += 1
            else:
                i = m.end()
    return hist


@dataclass
class Vocabulary:
    """Token <-> id mapping.

    Ordering rule: specials first at fixed low ids, then corpus tokens sorted by
    (descending frequency, ascending lexicographic). The lexicographic tiebreak
    is what makes construction deterministic -- frequency alone leaves ties
    resolved by dict insertion order, which varies with corpus row order.
    """

    itos: list[str] = field(default_factory=list)
    stoi: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.itos and not self.stoi:
            self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @classmethod
    def build(cls, texts: list[str], *, min_frequency: int = 1) -> Vocabulary:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text, on_unknown="char"))
        for special in SPECIAL_TOKENS:
            counts.pop(special, None)

        ordered = sorted(
            (t for t, c in counts.items() if c >= min_frequency),
            key=lambda t: (-counts[t], t),
        )
        itos = list(SPECIAL_TOKENS) + ordered
        return cls(itos=itos)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump({"itos": self.itos}, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls(itos=json.load(fh)["itos"])


class PSmilesTokenizer:
    """Tokenizer with an optional vocabulary.

    `tokenize`/`detokenize` work with no vocabulary at all -- they are pure
    grammar. `encode`/`decode` require one, and a vocabulary can only be built
    from data, so those paths stay unavailable until the dataset arrives.
    """

    def __init__(self, vocab: Vocabulary | None = None) -> None:
        self.vocab = vocab

    def tokenize(self, text: str, *, on_unknown: OnUnknown = "raise") -> list[str]:
        return tokenize(text, on_unknown=on_unknown)

    def detokenize(self, tokens: list[str]) -> str:
        return detokenize(tokens)

    def _require_vocab(self) -> Vocabulary:
        if self.vocab is None:
            raise ValueError(
                "This tokenizer has no vocabulary. Build one from the training "
                "split with Vocabulary.build(texts), or load a saved one. "
                "tokenize()/detokenize() do not require a vocabulary."
            )
        return self.vocab

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        on_unknown: OnUnknown = "raise",
    ) -> list[int]:
        vocab = self._require_vocab()
        tokens = self.tokenize(text, on_unknown=on_unknown)
        if add_special_tokens:
            tokens = [BOS_TOKEN, *tokens, EOS_TOKEN]
        return [vocab.stoi.get(t, vocab.unk_id) for t in tokens]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        vocab = self._require_vocab()
        tokens = [vocab.itos[i] for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return self.detokenize(tokens)
