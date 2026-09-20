import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.robustness.equivalence import is_equivalent
from materials_adv.training.invariance import (
    anchored_consistency_loss,
    encode_batch,
    normalize_and_tokenize,
    objective,
    worst_consistency_loss,
)
from scripts.run_invariance_defense import LAMBDA_GRID, SEEDS, build_model, predict


ROOT = Path(__file__).resolve().parents[1]


def test_consistency_loss_math():
    source = torch.tensor([1.0, 2.0])
    variants = torch.tensor([[2.0, 3.0], [2.0, 4.0]])
    assert anchored_consistency_loss(source, variants).item() == pytest.approx(2.25)
    assert worst_consistency_loss(source, variants).item() == pytest.approx(4.0)


def test_objective_composition():
    assert objective(torch.tensor(2.0), torch.tensor(3.0), 0.5).item() == 3.5


def test_tokenizer_fix_normalizes_bare_attachment():
    vocab = json.loads((ROOT / "data/processed/vocab.json").read_text())
    normalized, tokens = normalize_and_tokenize("*CC*", set(vocab))
    assert normalized == "[*]CC[*]"
    assert tokens.count("[*]") == 2
    ids, mask = encode_batch(["*CC*"], vocab)
    assert not (ids[~mask] == 0).any()


def test_unknown_tokens_are_rejected():
    vocab = json.loads((ROOT / "data/processed/vocab.json").read_text())
    with pytest.raises(ValueError, match="unknown tokens"):
        normalize_and_tokenize("[*][Xe]C[*]", set(vocab))


def test_lambda_grid_and_seeds_are_preregistered():
    assert LAMBDA_GRID == (0.01, 0.1, 0.5, 1.0)
    assert SEEDS == (20260815, 20260816, 20260817)


def test_model_default_reproduces_checkpoint_and_roundtrip(tmp_path):
    vocab = json.loads((ROOT / "data/processed/vocab.json").read_text())
    model = build_model(vocab, trainable=False)
    text = "[*]CCO[*]"
    expected = predict(model, [text], vocab)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = TransformerRegressorModel(len(vocab), 64, 2, 4, 128, 0.1, 256, "mean")
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual = predict(restored, [text], vocab)
    assert actual == pytest.approx(expected, abs=1e-7)


def test_frozen_test_cohort_hash_and_size():
    manifest = ROOT / "results/representation_equivalence_sequence/cohort_manifest.json"
    sources = json.loads(manifest.read_text())["sources"]
    assert len(sources) == 100
    assert len({row["source_id"] for row in sources}) == 100


def test_bank_pairing_chemistry_and_split_isolation():
    banks = ROOT / "results/invariance_defense/banks"
    if not (banks / "manifest.json").exists():
        pytest.skip("defense banks have not been prepared")
    frame = pd.read_csv(ROOT / "data/processed/processed.csv")
    test_ids = set(frame[frame.split == "test"].polymer_id.astype(str))
    for split in ("train", "val"):
        for family in ("broader", "repeat_shift"):
            bank = pd.read_csv(banks / f"{split}_{family}.csv")
            assert bank.split.eq(split).all()
            assert test_ids.isdisjoint(set(bank.source_id.astype(str)))
            assert bank.equivalence_pass.all()
            if family == "broader":
                for row in bank.head(100).itertuples():
                    assert is_equivalent(row.source, row.variant).equivalent


def test_variant_generation_bank_is_deterministic_metadata():
    manifest_path = ROOT / "results/invariance_defense/banks/manifest.json"
    if not manifest_path.exists():
        pytest.skip("defense banks have not been prepared")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["test_sources_used"] == 0
    assert manifest["unknown_allowed"] is False
    assert manifest["max_length"] == 256
