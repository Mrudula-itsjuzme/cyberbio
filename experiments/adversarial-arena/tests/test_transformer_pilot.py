from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from arena.contracts import CandidateRecord, SourceSample
from arena.transpolymer import TransPolymerDefender


ROOT = Path(__file__).parents[3] / "materials-adversarial"


def test_transformer_adapter_loads_canonical_checkpoint_and_predicts():
    defender = TransPolymerDefender(ROOT, seed=11,
        checkpoint_dir="results/models/transformer_regressor", epochs=1)
    predictions = defender.predict(["[*]CC[*]", "[*]CO[*]"])
    assert len(predictions) == 2
    assert all(isinstance(value, float) for value in predictions)
    assert defender.provenance()["sha256"]


def test_transformer_update_uses_only_representation_preserving_hard_examples(tmp_path, monkeypatch):
    defender = TransPolymerDefender(ROOT, seed=12,
        checkpoint_dir="results/models/transformer_regressor", epochs=1, batch_size=2)
    clean = [SourceSample("a", "[*]CC[*]", 4.0), SourceSample("b", "[*]CO[*]", 4.5)]
    monkeypatch.setattr(defender, "_validation_samples", lambda: clean)
    hard = [
        CandidateRecord("a", "[*]CC[*]", "C([*])C[*]", "RandomSMILES", "randomization", 0,
                        validity_status="valid", true_target=4.0),
        CandidateRecord("b", "[*]CO[*]", "[*]CN[*]", "Evolutionary", "substitution", 0,
                        validity_status="valid", true_target=4.5),
    ]
    before = {name: tensor.detach().clone() for name, tensor in defender.model.state_dict().items()}
    metrics = defender.train(clean, hard, round_id=1)
    assert metrics["representation_hard_example_count"] == 1
    assert metrics["diagnostic_only_chemistry_changing_count"] == 1
    assert metrics["loss_terms"]["chemistry_changing_supervised"] == 0.0
    assert any(not torch.equal(before[name], tensor) for name, tensor in defender.model.state_dict().items())
    checkpoint = tmp_path / "D1.pt"
    defender.checkpoint(str(checkpoint))
    restored = TransPolymerDefender(ROOT, seed=12,
        checkpoint_dir="results/models/transformer_regressor", epochs=1)
    restored.load_checkpoint(str(checkpoint))
    assert restored.checkpoint_id == "transformer-D001"
