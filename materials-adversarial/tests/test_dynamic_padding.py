"""Regression tests for the Phase 4 dynamic-padding optimization."""

import torch

from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.training.train import trim_padded_batch
from materials_adv.training.train import PolymerDataset
from materials_adv.data.scaler import TargetScaler
import pandas as pd


def _rows():
    return [
        {"ids": torch.tensor([1, 2, 3, 0, 0, 0]),
         "mask": torch.tensor([False, False, False, True, True, True]),
         "target": torch.tensor(1.0)},
        {"ids": torch.tensor([2, 3, 0, 0, 0, 0]),
         "mask": torch.tensor([False, False, True, True, True, True]),
         "target": torch.tensor(2.0)},
    ]


def test_trim_preserves_real_tokens_masks_and_targets():
    batch = trim_padded_batch(_rows())
    assert batch["ids"].tolist() == [[1, 2, 3], [2, 3, 0]]
    assert batch["mask"].tolist() == [
        [False, False, False], [False, False, True]
    ]
    assert batch["target"].tolist() == [1.0, 2.0]


def test_trimmed_and_fixed_width_predictions_are_numerically_equivalent():
    torch.manual_seed(17)
    model = TransformerRegressorModel(
        vocab_size=4, d_model=8, n_layers=1, n_heads=2,
        dim_feedforward=16, dropout=0.0, max_seq_len=256, pooling="mean",
    ).eval()
    fixed = torch.utils.data.default_collate(_rows())
    trimmed = trim_padded_batch(_rows())
    with torch.no_grad():
        fixed_predictions = model(fixed["ids"], fixed["mask"])
        trimmed_predictions = model(trimmed["ids"], trimmed["mask"])
    torch.testing.assert_close(trimmed_predictions, fixed_predictions, rtol=1e-6, atol=1e-7)


def test_polymer_dataset_precomputes_without_changing_encoded_example():
    scaler = TargetScaler()
    scaler.fit(pd.Series([1.0, 2.0]).to_numpy())
    data = pd.DataFrame({"psmiles": ["[*]CC[*]"], "target": [1.5]})
    dataset = PolymerDataset(data, ["[*]", "C"], 8, scaler)
    first = dataset[0]
    assert first["ids"].tolist() == [1, 2, 2, 1, 0, 0, 0, 0]
    assert first["mask"].tolist() == [False] * 4 + [True] * 4
    assert float(first["target"]) == 0.0
