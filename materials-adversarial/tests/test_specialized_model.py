"""Unit tests for TwoBranchTransformerRegressorModel and training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from materials_adv.data.scaler import TargetScaler
from materials_adv.models.specialized_transformer import (
    TwoBranchTransformerRegressor,
    TwoBranchTransformerRegressorModel,
)
from scripts.train_specialized_model import compute_auxiliary_losses, generate_training_pairs


def test_specialized_model_tensor_shapes():
    model = TwoBranchTransformerRegressorModel(vocab_size=36, d_model=64, branch_dim=32)
    src = torch.randint(1, 35, (4, 16))
    mask = torch.zeros((4, 16), dtype=torch.bool)

    pred, z_repr, z_chem = model.forward_branches(src, padding_mask=mask)

    assert pred.shape == (4,)
    assert z_repr.shape == (4, 32)
    assert z_chem.shape == (4, 32)


def test_deterministic_forward_pass_eval():
    model = TwoBranchTransformerRegressorModel(vocab_size=36, d_model=64, branch_dim=32)
    model.eval()
    src = torch.randint(1, 35, (2, 20))

    out1 = model(src)
    out2 = model(src)

    torch.testing.assert_close(out1, out2)


def test_gradient_flow_both_branches():
    model = TwoBranchTransformerRegressorModel(vocab_size=36, d_model=64, branch_dim=32)
    src = torch.randint(1, 35, (2, 10))

    pred, z_repr, z_chem = model.forward_branches(src)
    loss = pred.sum() + z_repr.sum() + z_chem.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients"


def test_invariance_loss_identical_vectors():
    z1 = torch.randn(4, 32)
    z2 = z1.clone()  # Identical

    loss_repr, _, _ = compute_auxiliary_losses(z1, torch.zeros_like(z1), z2, torch.zeros_like(z1))

    assert torch.isclose(loss_repr, torch.tensor(0.0), atol=1e-5)


def test_chemistry_margin_loss_behavior():
    z_clean = torch.zeros(2, 32)

    # Case A: Far apart (norm = 1.0 > margin 0.5) -> loss should be 0
    z_far = torch.ones(2, 32) / np.sqrt(32)
    _, loss_chem_far, _ = compute_auxiliary_losses(
        z_clean, z_clean, z_clean, z_far, chem_margin=0.5
    )
    assert torch.isclose(loss_chem_far, torch.tensor(0.0), atol=1e-5)

    # Case B: Identical (norm = 0.0 < margin 0.5) -> loss should be margin 0.5
    z_same = torch.zeros(2, 32)
    _, loss_chem_same, _ = compute_auxiliary_losses(
        z_clean, z_clean, z_clean, z_same, chem_margin=0.5
    )
    assert torch.isclose(loss_chem_same, torch.tensor(0.5), atol=1e-5)


def test_diversity_loss_behavior():
    z_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z_b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # Orthogonal -> cos sim = 0

    _, _, loss_div = compute_auxiliary_losses(z_a, z_b, z_a, z_b)
    assert torch.isclose(loss_div, torch.tensor(0.0), atol=1e-5)


def test_training_pair_generation_constraints():
    clean_reps = ["[*]CCC(c1ccccc1)C([*])c1ccocc1", "[*]CC([*])c1ccccc1"]
    vocab = ["[*]", "C", "c", "1", "ccccc", "ccocc"]

    c_reps, r_reps, s_reps, counts = generate_training_pairs(clean_reps, vocab, seed=20260815)

    assert len(c_reps) == len(clean_reps)
    assert len(r_reps) == len(clean_reps)
    assert len(s_reps) == len(clean_reps)
    assert counts["total_sources"] == 2


def test_predictor_wrapper_branch_embeddings():
    vocab = ["[*]", "C", "c", "1", "(", ")"]
    scaler = TargetScaler()
    scaler.mean = 4.0
    scaler.std = 1.5
    model = TwoBranchTransformerRegressorModel(vocab_size=len(vocab), d_model=64, branch_dim=32)

    predictor = TwoBranchTransformerRegressor(model, vocab, scaler)

    preds, z_repr, z_chem = predictor.get_branch_embeddings(["[*]CC[*]"])

    assert len(preds) == 1
    assert z_repr.shape == (1, 32)
    assert z_chem.shape == (1, 32)
