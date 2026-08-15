"""Preprocessing: canonicalization, dedup, conflict handling. PENDING(dataset).

The canonicalization helper below is implementable now (it only needs RDKit), but
the preprocessing PIPELINE is blocked: it depends on column names, on the Tg unit
verdict, and above all on a human decision about conflicting duplicates.

Conflicting duplicates (same structure, different Tg) must NOT be auto-resolved.
Mean / keep-first / drop-all are three different scientific choices with
different consequences, and one likely cause of conflict is a Kelvin/Celsius unit
mix -- averaging those would manufacture a value that is wrong in both units.
"""

from __future__ import annotations

from ..utils.optional import has_rdkit, require
from ..utils.pending import PendingImplementation


def canonicalize(psmiles: str) -> str | None:
    """RDKit canonical form, or None if unparseable / RDKit unavailable.

    Needed for honest dedup: the same polymer can be written many ways, so naive
    string matching misses most duplicates and leaks structures across splits.

    Caveat: RDKit canonicalizes '*' as a dummy atom. It has no concept of a
    polymer repeat unit, so canonical equality is structural-graph equality, not
    polymer identity.
    """
    if not has_rdkit():
        return None
    chem = require("rdkit.Chem", "chem")
    from rdkit import RDLogger  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    mol = chem.MolFromSmiles(psmiles)
    if mol is None:
        return None
    return chem.MolToSmiles(mol, canonical=True)


import json
import logging
from pathlib import Path
import pandas as pd

from ..utils.config import load_config
from .tokenizer import tokenize

logger = logging.getLogger(__name__)

def build_processed_dataset(config_path: str = "configs/dataset.yaml"):
    cfg = load_config(config_path)
    raw_path = Path(cfg["raw_dir"]) / cfg["file"]
    
    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    
    rep_col = cfg["representation_column"]
    tgt_col = cfg["target_column"]
    
    df = df.dropna(subset=[rep_col, tgt_col])
    
    logger.info("Canonicalizing representations for deduplication...")
    df["canonical_smiles"] = df[rep_col].apply(canonicalize)
    df = df.dropna(subset=["canonical_smiles"])
    
    policy = cfg["duplicate_policy"]
    logger.info(f"Resolving duplicates with policy: {policy}")
    
    if policy == "drop_all":
        # Keep only representations that appear exactly once
        df = df.drop_duplicates(subset=["canonical_smiles"], keep=False)
    elif policy == "keep_first":
        df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    elif policy == "mean":
        df = df.groupby("canonical_smiles", as_index=False).agg({
            rep_col: "first",
            tgt_col: "mean"
        })
    else:
        raise ValueError(f"Unknown duplicate policy: {policy}")
        
    proc_dir = Path(cfg["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_path = proc_dir / "processed.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed dataset with {len(df)} rows to {out_path}")
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = set()
    for psmiles in df[rep_col]:
        vocab.update(tokenize(psmiles))
    
    vocab_path = proc_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(sorted(list(vocab)), f)
        
    logger.info(f"Saved vocabulary with {len(vocab)} tokens to {vocab_path}")
    return df
