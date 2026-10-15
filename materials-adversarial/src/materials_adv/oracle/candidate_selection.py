"""Deterministic shortlist selection for later independent physical-oracle work."""

from __future__ import annotations

from collections import Counter

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def _canonical(value: str) -> str | None:
    mol = Chem.MolFromSmiles(value)
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _similarity(a: str, b: str) -> float:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return float("nan")
    fa = AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(ma)
    fb = AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mb)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


def select_oracle_candidates(
    candidates: pd.DataFrame,
    *,
    n: int = 20,
    max_per_source: int = 2,
) -> pd.DataFrame:
    """Select valid, chemistry-changing, diverse candidates without using labels.

    The input must contain source/candidate strings, attack family and model drift.
    Selection is round-robin across attack families after sorting by drift, with a
    hard cap per source and canonical-structure deduplication.
    """
    required = {"source_smiles", "candidate_smiles", "attack_family", "model_drift_eV"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"missing shortlist columns: {sorted(missing)}")
    # Parse a bounded high-information pool rather than fingerprinting tens of
    # thousands of near-duplicate trajectories.  The bound is per attack/model
    # stratum and is fixed by ``n``, so selection remains deterministic.
    work = candidates.copy()
    if "target_model" not in work:
        work["target_model"] = "unspecified"
    work = work.sort_values("model_drift_eV", ascending=False, kind="mergesort")
    pool_size = max(100, n * 20)
    work = work.groupby(["attack_family", "target_model"], sort=True, group_keys=False).head(pool_size)
    work["source_canonical"] = work["source_smiles"].map(_canonical)
    work["candidate_canonical"] = work["candidate_smiles"].map(_canonical)
    work = work.dropna(subset=["source_canonical", "candidate_canonical", "model_drift_eV"])
    work = work[work["source_canonical"] != work["candidate_canonical"]]
    work["similarity"] = [
        _similarity(a, b) for a, b in zip(work["source_smiles"], work["candidate_smiles"])
    ]
    work["drift_stratum"] = work.groupby(["attack_family", "target_model"])["model_drift_eV"].transform(
        lambda x: pd.qcut(x.rank(method="first"), q=min(3, len(x)), labels=False, duplicates="drop")
    )
    work["similarity_stratum"] = work.groupby(["attack_family", "target_model"])["similarity"].transform(
        lambda x: pd.qcut(x.rank(method="first"), q=min(3, len(x)), labels=False, duplicates="drop")
    )
    work["selection_stratum"] = (
        work["attack_family"].astype(str) + "|" + work["target_model"].astype(str)
        + "|d" + work["drift_stratum"].astype(str)
        + "|s" + work["similarity_stratum"].astype(str)
    )
    work = work.sort_values(
        ["selection_stratum", "model_drift_eV", "candidate_canonical"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    queues = {stratum: group.to_dict("records") for stratum, group in work.groupby("selection_stratum", sort=True)}
    counts: Counter[str] = Counter()
    seen_candidates: set[str] = set()
    selected: list[dict] = []
    while len(selected) < n and any(queues.values()):
        progressed = False
        for family in sorted(queues):
            while queues[family]:
                row = queues[family].pop(0)
                source = row["source_canonical"]
                candidate = row["candidate_canonical"]
                if candidate in seen_candidates or counts[source] >= max_per_source:
                    continue
                selected.append(row)
                seen_candidates.add(candidate)
                counts[source] += 1
                progressed = True
                break
            if len(selected) == n:
                break
        if not progressed:
            break
    result = pd.DataFrame(selected)
    forbidden = {"true_label", "property_value", "bandgap"} & set(result.columns)
    if forbidden:
        result = result.drop(columns=sorted(forbidden))
    return result
