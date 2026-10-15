"""Fail-closed scientific-integrity checks for the active polyVERSE lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_active_lineage(root: Path, checkpoint: Path, scaler_path: Path) -> dict:
    df = pd.read_csv(root / "data/processed/processed.csv")
    splits = json.loads((root / "data/processed/splits.json").read_text())
    vocab = json.loads((root / "data/processed/vocab.json").read_text())
    required = {"train", "val", "test"}
    if not required.issubset(splits) or splits.get("test_sealed") is not True:
        raise AssertionError("split artifact is incomplete or test is not sealed")
    split_sets = {name: set(map(int, splits[name])) for name in required}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("row indices cross split boundaries")
    if set().union(*split_sets.values()) != set(range(len(df))):
        raise AssertionError("split indices do not cover the processed dataset exactly")

    canonical: dict[str, set[str]] = {}
    invalid = 0
    for name in sorted(required):
        identities: set[str] = set()
        for smi in df.iloc[splits[name]]["original_representation"].astype(str):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid += 1
                continue
            identities.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        canonical[name] = identities
    overlaps = {
        f"{a}_{b}": len(canonical[a] & canonical[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if invalid or any(overlaps.values()):
        raise AssertionError(f"invalid structures or canonical split leakage: invalid={invalid}, overlaps={overlaps}")
    if set(df["units"].astype(str)) != {"eV"} or set(df["source_dataset"].astype(str)) != {"polyverse_bandgap"}:
        raise AssertionError("dataset lineage or units mismatch")
    for name in required:
        expected = name if name != "val" else "val"
        if set(df.iloc[splits[name]]["split"].astype(str)) != {expected}:
            raise AssertionError(f"dataframe split column disagrees for {name}")

    scaler = TargetScaler.load(scaler_path)
    train_targets = df.iloc[splits["train"]]["property_value"].to_numpy(float)
    scaler_delta = {
        "mean": float(scaler.mean - np.mean(train_targets)),
        "std": float(scaler.std - np.std(train_targets)),
    }
    if abs(scaler_delta["mean"]) > 1e-12 or abs(scaler_delta["std"]) > 1e-12:
        raise AssertionError("scaler was not fit exactly on the frozen training targets")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    rows = int(state["embedding.weight"].shape[0])
    if rows != len(vocab) + 1:
        raise AssertionError(f"checkpoint/tokenizer mismatch: embedding rows={rows}, vocab+pad={len(vocab)+1}")
    vocab_set = set(vocab)
    oov = sorted({tok for smi in df["original_representation"].astype(str) for tok in tokenize(smi) if tok not in vocab_set})
    if oov:
        raise AssertionError(f"dataset contains out-of-vocabulary tokens: {oov}")
    return {
        "status": "passed",
        "dataset_rows": len(df),
        "split_counts": {name: len(splits[name]) for name in sorted(required)},
        "canonical_cross_split_overlaps": overlaps,
        "invalid_structures": invalid,
        "units": "eV",
        "source_dataset": "polyverse_bandgap",
        "scaler_train_delta": scaler_delta,
        "vocab_size": len(vocab),
        "checkpoint_embedding_rows": rows,
        "checkpoint_sha256": sha256(checkpoint),
        "scaler_sha256": sha256(scaler_path),
        "vocab_sha256": sha256(root / "data/processed/vocab.json"),
        "splits_sha256": sha256(root / "data/processed/splits.json"),
    }
