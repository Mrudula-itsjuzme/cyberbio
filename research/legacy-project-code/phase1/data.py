"""
data.py — Dataset module for Phase 1.

Generates a synthetic dataset of SMILES-like material sequences with a
compositional regression target property.  A real dataset loader stub is also
provided — drop in a CSV path and it will switch to real-data mode automatically.

Mode is printed clearly at runtime so the reader always knows which path is active.

Part of: Learning to Attack and Defend — Phase 1
"""

import os
import re
import random
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional

# ── SMILES token vocabulary (shared with representation.py) ──────────────────
# Ordered longest-first so multi-char tokens are matched before single chars.
SMILES_VOCAB: List[str] = [
    # Multi-char atoms
    "Cl", "Br", "Si", "Se", "Na", "Ca",
    # Single-char atoms
    "C", "N", "O", "S", "F", "P", "B", "I",
    # Aromatic atoms
    "c", "n", "o", "s",
    # Bond symbols
    "=", "#", "-", ":", ".",
    # Ring / branch / charge
    "(", ")", "[", "]", "+",
    # Ring closure digits
    "1", "2", "3", "4", "5", "6",
    # Hydrogen notation
    "H",
]

# Atom tokens only (for generating backbones and for plausibility checks)
ATOM_TOKENS: List[str] = [
    "C", "N", "O", "S", "F", "Cl", "Br", "c", "n", "o", "s", "P", "B", "I",
]
BRANCH_TOKENS: List[str] = ["(", ")"]
BOND_TOKENS:   List[str] = ["=", "#", "-"]
RING_DIGITS:   List[str] = ["1", "2", "3", "4", "5", "6"]

# ── Property weights (kept fixed for reproducibility) ───────────────────────
# Synthetic property: weighted linear combination of atom counts + ring count
_PROPERTY_WEIGHTS: Dict[str, float] = {
    "C": 0.10, "c": 0.12, "N": 0.18, "n": 0.20, "O": 0.25, "o": 0.22,
    "S": 0.30, "s": 0.28, "F": 0.35, "Cl": 0.40, "Br": 0.45,
    "P": 0.22, "B": 0.15, "I": 0.50, "Si": 0.20, "Se": 0.38,
    "=": 0.08, "#": 0.15,
}
_RING_WEIGHT = 0.20


# ── Synthetic sequence generator ─────────────────────────────────────────────

def _open_ring(seq: List[str], open_rings: List[str]) -> List[str]:
    """Possibly open a ring closure at the current position."""
    available = [d for d in RING_DIGITS if d not in open_rings]
    if available and random.random() < 0.15:
        digit = random.choice(available)
        open_rings.append(digit)
        seq.append(digit)
    return seq


def _close_ring(seq: List[str], open_rings: List[str]) -> List[str]:
    """Possibly close an open ring at the current position."""
    if open_rings and random.random() < 0.40:
        digit = open_rings.pop(0)
        seq.append(digit)
    return seq


def _generate_one_smiles(min_atoms: int = 4,
                          max_atoms: int = 14,
                          rng: Optional[random.Random] = None) -> str:
    """
    Generate a single synthetic SMILES-like string.

    The string is constructed as a sequence of randomly chosen atoms with
    occasional bond annotations, branches, and ring-closure digits.
    It is not guaranteed to be chemically valid (RDKit not used) but obeys
    the basic structural grammar checked by the plausibility filter.
    """
    if rng is None:
        rng = random.Random()

    n_atoms = rng.randint(min_atoms, max_atoms)
    tokens: List[str] = []
    open_rings: List[str] = []
    branch_depth = 0

    for i in range(n_atoms):
        # Occasionally add a bond token between atoms
        if tokens and rng.random() < 0.20:
            tokens.append(rng.choice(BOND_TOKENS))

        # Possibly open a branch
        if branch_depth < 2 and rng.random() < 0.12:
            tokens.append("(")
            branch_depth += 1

        # Add an atom
        atom = rng.choice(ATOM_TOKENS[:10])   # bias toward common atoms
        tokens.append(atom)

        # Possibly annotate ring closure
        _open_ring(tokens, open_rings)
        _close_ring(tokens, open_rings)

        # Possibly close a branch
        if branch_depth > 0 and rng.random() < 0.35:
            tokens.append(")")
            branch_depth -= 1

    # Force-close any still-open branches/rings at end
    for _ in range(branch_depth):
        tokens.append(")")
    for digit in open_rings:
        tokens.append(digit)

    return "".join(tokens)


def _compute_property(smiles: str, noise_scale: float = 0.05,
                       rng: Optional[np.random.Generator] = None) -> float:
    """
    Compute a synthetic regression target for a SMILES string.

    The property is a weighted sum of token occurrences (plus a small ring
    bonus and Gaussian noise), normalised to roughly [0, 5].
    """
    if rng is None:
        rng = np.random.default_rng()

    # Count occurrences of each weighted token
    prop = 0.0
    for token, weight in _PROPERTY_WEIGHTS.items():
        prop += smiles.count(token) * weight

    # Ring bonus: each matched ring-closure pair adds a bonus
    for digit in RING_DIGITS:
        count = smiles.count(digit)
        prop += (count // 2) * _RING_WEIGHT

    # Additive noise
    prop += rng.normal(0.0, noise_scale)
    return float(np.clip(prop, 0.0, 10.0))


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = 2000,
    min_atoms: int = 4,
    max_atoms: int = 14,
    noise_scale: float = 0.05,
    seed: int = 42,
) -> Tuple[List[str], List[float]]:
    """
    Generate a synthetic SMILES/property dataset.

    Returns:
        (smiles_list, property_list) — parallel lists of strings and float values.
    """
    print("[DATA MODE] -- SYNTHETIC --")
    print(f"  Generating {n_samples} synthetic SMILES-like sequences (seed={seed})")

    rng_py  = random.Random(seed)
    rng_np  = np.random.default_rng(seed)

    smiles_list:   List[str]   = []
    property_list: List[float] = []
    seen_hashes = set()

    attempts = 0
    while len(smiles_list) < n_samples:
        attempts += 1
        smi  = _generate_one_smiles(min_atoms, max_atoms, rng=rng_py)
        # Deduplication via hash
        h = hashlib.md5(smi.encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Basic validity: length >= 4 tokens
        if len(smi) < 4:
            continue

        prop = _compute_property(smi, noise_scale=noise_scale, rng=rng_np)
        smiles_list.append(smi)
        property_list.append(prop)

    print(f"  Generated {len(smiles_list)} unique sequences in {attempts} attempts.")
    return smiles_list, property_list


# ── Real-data loader stub ────────────────────────────────────────────────────

def load_real_dataset(csv_path: str,
                       smiles_col: str = "smiles",
                       target_col: str = "property") -> Tuple[List[str], List[float]]:
    """
    Load a real SMILES-property dataset from a CSV file.

    Args:
        csv_path:   path to the CSV.
        smiles_col: column name containing SMILES strings.
        target_col: column name containing the regression target.

    Returns:
        (smiles_list, property_list)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load a real CSV dataset.")

    print(f"[DATA MODE] -- REAL DATA -- {csv_path}")
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{smiles_col}' and '{target_col}'. "
                         f"Found: {list(df.columns)}")
    smiles_list   = df[smiles_col].astype(str).tolist()
    property_list = df[target_col].astype(float).tolist()
    print(f"  Loaded {len(smiles_list)} rows from {csv_path}.")
    return smiles_list, property_list


# ── Cleaning ─────────────────────────────────────────────────────────────────

def clean_dataset(smiles_list:   List[str],
                  property_list: List[float],
                  min_length:    int   = 4,
                  max_length:    int   = 200,
                  deduplicate:   bool  = True) -> Tuple[List[str], List[float]]:
    """
    Filter and deduplicate a SMILES/property dataset.

    Filters applied:
      1. Remove entries shorter than min_length or longer than max_length characters.
      2. Remove entries with NaN property values.
      3. Deduplicate by exact SMILES string.

    Returns:
        Cleaned (smiles_list, property_list).
    """
    print(f"[Clean] Before cleaning: {len(smiles_list)} samples")
    seen = set()
    clean_smi, clean_prop = [], []

    for smi, prop in zip(smiles_list, property_list):
        if not isinstance(smi, str):
            continue
        if not (min_length <= len(smi) <= max_length):
            continue
        if not np.isfinite(prop):
            continue
        if deduplicate:
            if smi in seen:
                continue
            seen.add(smi)
        clean_smi.append(smi)
        clean_prop.append(prop)

    print(f"[Clean] After cleaning:  {len(clean_smi)} samples "
          f"(removed {len(smiles_list) - len(clean_smi)})")
    return clean_smi, clean_prop


# ── Train / val / test split ─────────────────────────────────────────────────

def split_dataset(
    smiles_list:   List[str],
    property_list: List[float],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 42,
) -> Dict[str, Tuple[List[str], List[float]]]:
    """
    Random stratified split into train / val / test sets.

    Returns:
        {
            "train": (smiles, props),
            "val":   (smiles, props),
            "test":  (smiles, props),
        }
    """
    assert abs(train_frac + val_frac - 0.85) < 1e-6 or \
           train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1.0"

    n = len(smiles_list)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    def _pick(idx_list):
        s = [smiles_list[i] for i in idx_list]
        p = [property_list[i] for i in idx_list]
        return s, p

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    splits = {
        "train": _pick(train_idx),
        "val":   _pick(val_idx),
        "test":  _pick(test_idx),
    }

    for name, (s, p) in splits.items():
        print(f"[Split] {name:5s}: {len(s):5d} samples  "
              f"| property range [{min(p):.3f}, {max(p):.3f}]  "
              f"| mean {np.mean(p):.3f}")

    return splits


# ── Dataset stats ─────────────────────────────────────────────────────────────

def dataset_stats(smiles_list: List[str], property_list: List[float]) -> Dict:
    """Print and return basic statistics about the dataset."""
    lengths = [len(s) for s in smiles_list]
    props   = np.array(property_list)
    stats   = {
        "n_samples":     len(smiles_list),
        "len_mean":      float(np.mean(lengths)),
        "len_min":       int(np.min(lengths)),
        "len_max":       int(np.max(lengths)),
        "prop_mean":     float(np.mean(props)),
        "prop_std":      float(np.std(props)),
        "prop_min":      float(np.min(props)),
        "prop_max":      float(np.max(props)),
    }
    print("\n-- Dataset Statistics ----------------------------------")
    for k, v in stats.items():
        print(f"  {k:<16}: {v:.4f}" if isinstance(v, float) else f"  {k:<16}: {v}")
    print("--------------------------------------------------------\n")
    return stats


# ── Convenience: full pipeline ───────────────────────────────────────────────

def load_or_generate(
    csv_path:    Optional[str] = None,
    smiles_col:  str           = "smiles",
    target_col:  str           = "property",
    n_synthetic: int           = 2000,
    seed:        int           = 42,
) -> Dict[str, Tuple[List[str], List[float]]]:
    """
    Master entry point: load real data if csv_path is supplied, otherwise generate synthetic.
    Cleans and splits the data and returns the split dict.
    """
    if csv_path and os.path.isfile(csv_path):
        raw_smi, raw_prop = load_real_dataset(csv_path, smiles_col, target_col)
    else:
        if csv_path:
            print(f"[WARNING] csv_path='{csv_path}' not found — falling back to synthetic data.")
        raw_smi, raw_prop = generate_synthetic_dataset(n_samples=n_synthetic, seed=seed)

    clean_smi, clean_prop = clean_dataset(raw_smi, raw_prop)
    dataset_stats(clean_smi, clean_prop)
    splits = split_dataset(clean_smi, clean_prop, seed=seed)
    return splits


# ── __main__ quick-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    splits = load_or_generate(n_synthetic=500, seed=42)
    print("\nSample training SMILES:")
    for s, p in zip(*splits["train"])[:5]:
        print(f"  {s:<25}  →  property = {p:.4f}")
    print("\n[data.py] Quick-test complete.")
