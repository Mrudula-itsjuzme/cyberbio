"""
representation.py — Tokenizer and encoder for SMILES/formula sequences.

Self-contained: no RDKit or external cheminformatics dependency.
Uses a curated SMILES-aware vocabulary with multi-char token priority.

Part of: Learning to Attack and Defend — Phase 1
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional

# ── Vocabulary definition ─────────────────────────────────────────────────────

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Chemistry tokens — ordered longest-first for greedy tokenisation
CHEMISTRY_TOKENS: List[str] = [
    # Two-char atoms (must come before single chars)
    "Cl", "Br", "Si", "Se", "Na", "Ca", "Al", "Mg",
    # Single-char atoms (uppercase = aliphatic, lowercase = aromatic)
    "C", "N", "O", "S", "F", "P", "B", "I",
    "c", "n", "o", "s", "p",
    # Bond symbols
    "=", "#", "-", ":", ".",
    # Branches / charged brackets
    "(", ")", "[", "]", "+",
    # Ring closure digits
    "1", "2", "3", "4", "5", "6", "7", "8",
    # Common inside-bracket modifiers (treated as single tokens)
    "H", "@", "/", "\\",
]

FULL_VOCAB: List[str] = SPECIAL_TOKENS + CHEMISTRY_TOKENS

# Sorted longest-first for the tokeniser's greedy regex
_SORTED_CHEM = sorted(CHEMISTRY_TOKENS, key=len, reverse=True)
_TOKEN_PATTERN = re.compile(
    r"(" + "|".join(re.escape(t) for t in _SORTED_CHEM) + r")"
)


# ── Vocabulary class ──────────────────────────────────────────────────────────

class Vocabulary:
    """
    Bidirectional token ↔ integer mapping with special-token handling.

    Attributes:
        token_to_id: dict mapping token string → integer id.
        id_to_token: dict mapping integer id → token string.
        pad_id, unk_id, bos_id, eos_id: integer ids for special tokens.
        size: total vocabulary size.
    """

    def __init__(self, tokens: List[str] = None):
        if tokens is None:
            tokens = FULL_VOCAB
        self._tokens = list(tokens)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self._tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self._tokens)}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]

    @property
    def size(self) -> int:
        return len(self._tokens)

    def encode_token(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def decode_id(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, UNK_TOKEN)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f"Vocabulary(size={self.size}, "
                f"pad={self.pad_id}, unk={self.unk_id}, "
                f"bos={self.bos_id}, eos={self.eos_id})")


# ── Tokeniser ─────────────────────────────────────────────────────────────────

class SMILESTokenizer:
    """
    Character / rule-based tokeniser for SMILES-like material sequences.

    Tokenisation strategy:
      1. Greedy left-to-right matching against CHEMISTRY_TOKENS (longest match first).
      2. Any unmatched character → <UNK>.
      3. Optionally wrap with <BOS> / <EOS>.

    No RDKit dependency — works on synthetic sequences that may not be
    chemically valid but conform to the SMILES-like grammar.
    """

    def __init__(self, vocab: Vocabulary = None, add_special: bool = True):
        self.vocab       = vocab if vocab is not None else Vocabulary()
        self.add_special = add_special

    # ── Tokenise ──────────────────────────────────────────────────────────────

    def tokenize(self, smiles: str) -> List[str]:
        """
        Split a SMILES string into a list of token strings.

        Args:
            smiles: a SMILES or SMILES-like string.

        Returns:
            List of token strings, with <BOS>/<EOS> if add_special=True.
        """
        tokens: List[str] = []
        if self.add_special:
            tokens.append(BOS_TOKEN)

        pos = 0
        while pos < len(smiles):
            m = _TOKEN_PATTERN.match(smiles, pos)
            if m:
                tokens.append(m.group(0))
                pos = m.end()
            else:
                # Unrecognised character — map to <UNK>
                tokens.append(UNK_TOKEN)
                pos += 1

        if self.add_special:
            tokens.append(EOS_TOKEN)
        return tokens

    def encode(self, smiles: str) -> List[int]:
        """Tokenize and convert to integer IDs."""
        return [self.vocab.encode_token(t) for t in self.tokenize(smiles)]

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """
        Convert a list of integer IDs back to a SMILES string.

        Args:
            ids:           list of integer token IDs.
            strip_special: if True, remove <PAD>/<BOS>/<EOS>/<UNK> from output.
        """
        special_ids = {self.vocab.pad_id, self.vocab.bos_id,
                       self.vocab.eos_id, self.vocab.unk_id}
        tokens = []
        for i in ids:
            t = self.vocab.decode_id(i)
            if strip_special and i in special_ids:
                continue
            tokens.append(t)
        return "".join(tokens)

    def decode_tokens(self, tokens: List[str], strip_special: bool = True) -> str:
        """Convert a list of token strings back to a SMILES string."""
        special_set = set(SPECIAL_TOKENS)
        out = []
        for t in tokens:
            if strip_special and t in special_set:
                continue
            out.append(t)
        return "".join(out)

    # ── Batch utilities ───────────────────────────────────────────────────────

    def encode_batch(self, smiles_list: List[str],
                     max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a list of SMILES strings into a padded integer array.

        Args:
            smiles_list: list of SMILES strings.
            max_len:     maximum sequence length (auto-inferred if None).

        Returns:
            ids_array   — np.ndarray of shape (N, max_len), int64, padded with pad_id.
            lengths     — np.ndarray of shape (N,), int64, actual length of each sequence.
        """
        encoded = [self.encode(s) for s in smiles_list]
        lengths = [len(e) for e in encoded]

        if max_len is None:
            max_len = max(lengths)
        else:
            max_len = min(max_len, max(lengths))

        pad_id = self.vocab.pad_id
        ids_array = np.full((len(encoded), max_len), pad_id, dtype=np.int64)
        for i, e in enumerate(encoded):
            l = min(len(e), max_len)
            ids_array[i, :l] = e[:l]

        return ids_array, np.array(lengths, dtype=np.int64)

    def token_counts(self, smiles_list: List[str]) -> Dict[str, int]:
        """Count token frequency across a corpus (useful for transition matrix)."""
        from collections import Counter
        counter: Counter = Counter()
        for smi in smiles_list:
            for t in self.tokenize(smi):
                counter[t] += 1
        return dict(counter)


# ── Encoding utilities (functional API) ──────────────────────────────────────

def build_default_tokenizer(add_special: bool = True) -> SMILESTokenizer:
    """Convenience factory for the default tokenizer + vocabulary."""
    vocab = Vocabulary(FULL_VOCAB)
    return SMILESTokenizer(vocab, add_special=add_special)


def pad_sequence(ids: List[int], max_len: int, pad_id: int = 0) -> List[int]:
    """Right-pad or truncate a list of ids to max_len."""
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [pad_id] * (max_len - len(ids))


def make_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """
    Create a boolean padding mask for use with PyTorch Transformer.

    Returns:
        mask of shape (N, max_len) where True = padded position (to ignore).
    """
    mask = np.ones((len(lengths), max_len), dtype=bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = False   # False = valid (attend to this position)
    return mask


# ── Positional encoding (NumPy, for reference / non-PyTorch contexts) ─────────

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Compute sinusoidal positional encoding matrix.

    Returns:
        pe: np.ndarray of shape (max_len, d_model).
    """
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(max_len)[:, None]         # (max_len, 1)
    div = np.exp(
        np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )                                           # (d_model/2,)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


# ── __main__ quick-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = build_default_tokenizer()
    vocab     = tokenizer.vocab

    print(repr(vocab))
    print(f"Vocabulary tokens: {vocab._tokens[:10]} ... [{vocab.size} total]")

    test_smiles = [
        "CCO",
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin-like
        "C1CCCCC1",                 # Cyclohexane-like
        "ClCCBr",
        "C#N",
    ]

    print("\n── Tokenisation examples ───────────────────────────────────")
    for smi in test_smiles:
        tokens = tokenizer.tokenize(smi)
        ids    = tokenizer.encode(smi)
        recon  = tokenizer.decode(ids)
        print(f"  Input:  {smi}")
        print(f"  Tokens: {tokens}")
        print(f"  IDs:    {ids}")
        print(f"  Recon:  {recon}")
        print()

    print("── Batch encoding ─────────────────────────────────────────")
    ids_arr, lengths = tokenizer.encode_batch(test_smiles)
    print(f"  Shape: {ids_arr.shape}  |  Lengths: {lengths}")

    mask = make_padding_mask(lengths, max_len=ids_arr.shape[1])
    print(f"  Padding mask shape: {mask.shape}")

    pe = sinusoidal_positional_encoding(max_len=50, d_model=64)
    print(f"  Positional encoding shape: {pe.shape}")

    print("\n[representation.py] Quick-test complete.")
