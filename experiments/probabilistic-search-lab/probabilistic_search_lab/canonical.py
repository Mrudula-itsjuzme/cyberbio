from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parents[1]
MATERIALS_ROOT = REPO_ROOT / "materials-adversarial"
COMPARISON_ROOT = REPO_ROOT / "comparison-experiments"
sys.path.insert(0, str(MATERIALS_ROOT / "src"))

from materials_adv.domain.chemistry.plausibility import ChemicalPlausibilityValidator
from materials_adv.framework.predictors import GraphMPNNPredictor
from materials_adv.validation.pipeline import validate as canonical_validate

from .interfaces import Validation


class CanonicalValidator:
    """Preserve tri-state representation checks and source-relative constraints."""

    def __init__(self, min_tanimoto_similarity: float = 0.5):
        self.domain = ChemicalPlausibilityValidator(
            min_tanimoto_similarity=min_tanimoto_similarity,
            require_attachment_preservation=True,
            enforce_valence=True,
            enforce_mw_bounds=True,
        )

    def validate(self, source: str, candidate: str) -> Validation:
        canonical = canonical_validate(candidate, check_plausibility=True)
        if canonical.representation_valid is not True:
            return Validation(canonical.representation_valid, None, canonical.rejection_reasons)
        passed, details = self.domain.validate(source, candidate)
        reasons = list(canonical.rejection_reasons)
        if not canonical.plausible:
            reasons.extend(key for key, value in canonical.plausibility_flags.items() if not value)
        if not passed:
            reasons.extend(key for key, value in details.items() if isinstance(value, bool) and not value)
        constraint_pass = bool(canonical.plausible and passed)
        return Validation(True, constraint_pass, tuple(dict.fromkeys(reasons)))


def load_model(device: str = "cpu") -> GraphMPNNPredictor:
    return GraphMPNNPredictor(
        checkpoint=MATERIALS_ROOT / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt",
        scaler_path=MATERIALS_ROOT / "results/models/transformer_regressor/scaler.json",
        device=device,
        strict=True,
    )


def load_sources(split: str, limit: int) -> list[tuple[str, str]]:
    import pandas as pd

    frame = pd.read_csv(MATERIALS_ROOT / "data/processed/processed.csv")
    frozen = json.loads((COMPARISON_ROOT / "data/frozen_source_ids.json").read_text())
    rows = []
    for index in frozen["source_ids"]:
        row = frame.iloc[int(index)]
        if str(row["split"]) != split:
            continue
        rows.append((str(row["polymer_id"]), str(row["original_representation"])))
        if len(rows) == limit:
            break
    if len(rows) < limit:
        raise ValueError(f"requested {limit} frozen {split!r} sources, found {len(rows)}")
    return rows
