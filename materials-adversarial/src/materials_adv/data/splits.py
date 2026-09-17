"""Train/validation/test splitting.

Design rule: splits are written to disk as an explicit artifact and read back, never re-shuffled at runtime.
That freezes membership across downstream experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.optional import has_rdkit
from ..utils.pending import PendingImplementation
from ..utils.config import load_config

logger = logging.getLogger(__name__)


def grouped_split(
    group_keys: Sequence[str],
    *,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict[str, list[int]]:
    """Split indices so that no group key spans two splits.

    Generic over the grouping key: pass canonical SMILES for dedup-safe splitting,
    or scaffold keys for structure-aware splitting. Proportional group balancing
    ensures all splits receive non-zero allocations even with large group variations.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid fractions: train={train_frac}, val={val_frac}")

    groups: dict[str, list[int]] = {}
    for i, key in enumerate(group_keys):
        groups.setdefault(key, []).append(i)

    ordered = sorted(groups, key=lambda k: len(groups[k]), reverse=True)

    n_total = len(group_keys)
    n_train_target = int(round(train_frac * n_total))
    n_val_target = int(round(val_frac * n_total))
    n_test_target = n_total - n_train_target - n_val_target

    targets = {"train": n_train_target, "val": n_val_target, "test": n_test_target}
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for key in ordered:
        members = groups[key]
        # Pick split with lowest ratio of current_len / target_len
        ratios = {s: len(splits[s]) / max(targets[s], 1) for s in ["train", "val", "test"]}
        best_split = min(ratios, key=ratios.get)
        splits[best_split].extend(members)

    return {k: sorted(v) for k, v in splits.items()}


def scaffold_key(psmiles: str) -> str:
    """Bemis-Murcko scaffold, for structure-aware splitting."""
    if not has_rdkit():
        raise PendingImplementation(
            what="scaffold_key(): Bemis-Murcko scaffold extraction requires RDKit",
            blocked_on="rdkit",
            unblocks_when='RDKit is installed: pip install -e ".[chem]"',
        )
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        return ""


def build_splits(config_path: str = "configs/dataset.yaml"):
    cfg = load_config(config_path)
    proc_dir = Path(cfg["processed_dir"])
    proc_path = proc_dir / "processed.csv"

    if not proc_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {proc_path}. Run preprocessing first.")

    df = pd.read_csv(proc_path)
    rep_col = cfg["representation_column"]

    split_cfg = cfg["split"]
    strategy = split_cfg["strategy"]
    seed = split_cfg["seed"]
    train_frac = split_cfg["train_frac"]
    val_frac = split_cfg["val_frac"]

    logger.info(f"Building splits using strategy: {strategy}")

    if strategy == "scaffold":
        keys = df[rep_col].apply(scaffold_key).tolist()
    elif strategy == "random":
        keys = df["canonical_smiles"].tolist()  # canonical smiles guarantees dedup
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    splits = grouped_split(keys, seed=seed, train_frac=train_frac, val_frac=val_frac)

    artifact_path = Path(split_cfg["artifact"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with open(artifact_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        f"Saved splits to {artifact_path}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return splits
