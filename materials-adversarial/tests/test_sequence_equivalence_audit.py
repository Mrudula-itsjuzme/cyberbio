from __future__ import annotations

import ast, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from materials_adv.data.tokenizer import tokenize
from materials_adv.robustness.equivalence import generate_equivalents, is_equivalent

ROOT = Path(__file__).resolve().parents[1]


def test_known_equivalent_and_non_equivalent_pairs():
    assert is_equivalent("CCO", "OCC").equivalent
    assert not is_equivalent("CCO", "CCN").equivalent


def test_attachment_change_rejected():
    result = is_equivalent("[*]CC[*]", "[*]CCC[*]")
    assert not result.equivalent


@pytest.mark.parametrize("family", ["start_point_shift", "orientation_reversal", "randomized_atom_ordering"])
def test_requested_transformation_semantics(family):
    variants, _ = generate_equivalents("[*]CC(O)C[*]", seed=17, target=30)
    selected = [v for v in variants if v.family == family]
    if not selected:
        pytest.skip(f"no unique {family} serialization for this small graph")
    assert all(is_equivalent(v.source_representation, v.variant_representation).equivalent for v in selected)


def test_repeat_shift_tag_and_duplicate_removal_and_determinism():
    first, _ = generate_equivalents("[*]CC(O)CC[*]", seed=9, target=25)
    second, _ = generate_equivalents("[*]CC(O)CC[*]", seed=9, target=25)
    assert first == second
    strings = [v.variant_representation for v in first]
    assert len(strings) == len(set(strings))
    assert any(v.subset == "MIPS_REPEAT_SHIFT_COMPATIBLE" for v in first)


def test_canonical_tokenizer_is_used_by_runner():
    tree = ast.parse((ROOT / "scripts/run_sequence_equivalence_audit.py").read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert any(n.module == "materials_adv.data.tokenizer" and any(a.name == "tokenize" for a in n.names) for n in imports)


def test_checkpoint_load_and_historical_reproduction():
    from scripts.run_sequence_equivalence_audit import load_frozen_model, reproduce_history
    _, predictor, _ = load_frozen_model()
    frame = pd.read_csv(ROOT / "data/processed/processed.csv")
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    assert reproduce_history(predictor, frame, splits)["status"] == "PASS"


def test_source_metrics_formulas():
    source, variants = 1.0, np.array([0.0, 2.5, 1.5])
    all_values = np.r_[source, variants]
    assert np.max(np.abs(variants-source)) == 1.5
    assert np.max(np.abs(all_values-all_values.mean())) == 1.25


def test_no_optimizer_or_training_and_parameters_frozen():
    from scripts.run_sequence_equivalence_audit import load_frozen_model
    model, _, _ = load_frozen_model()
    assert not any(p.requires_grad for p in model.parameters())
    text = (ROOT / "scripts/run_sequence_equivalence_audit.py").read_text()
    assert "torch.optim" not in text and ".backward(" not in text


def test_no_train_split_leakage_and_pair_consistency_if_outputs_exist():
    out = ROOT / "results/representation_equivalence_sequence"
    if not (out / "cohort_manifest.json").exists(): pytest.skip("audit not run")
    manifest = json.loads((out / "cohort_manifest.json").read_text())
    assert {row["split"] for row in manifest["sources"]} == {"test"}
    variants = pd.read_csv(out / "equivalent_variants.csv")
    sources = {r["source_id"]: r["representation"] for r in manifest["sources"]}
    assert all(row.source_representation == sources[row.source_id] for row in variants.itertuples())
    assert variants.equivalent_pass.all()
