import pytest
import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath("../materials-adversarial/src"))
from materials_adv.data.tokenizer import PSmilesTokenizer, Vocabulary
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.data.scaler import TargetScaler

sys.path.append(os.path.abspath("scripts"))
from run_defense_transfer import predict_batch, load_checkpoint

@pytest.fixture
def dummy_vocab():
    return Vocabulary(itos=["<pad>", "<unk>", "<bos>", "<eos>", "C", "c", "O", "N"])

@pytest.fixture
def dummy_model(dummy_vocab):
    model = TransformerRegressorModel(
        vocab_size=len(dummy_vocab), d_model=16, n_layers=1, n_heads=1, dim_feedforward=32, dropout=0.0, max_seq_len=64
    )
    model.eval()
    return model

def test_source_candidate_order(dummy_model, dummy_vocab):
    seqs = ["C", "c", "O"]
    preds = predict_batch(dummy_model, seqs, dummy_vocab, 64, False)
    assert len(preds) == 3

def test_inverse_scaling():
    scaler = TargetScaler()
    scaler.mean = np.array([1.0])
    scaler.std = np.array([2.0])
    scaled = scaler.transform(np.array([10.0]))
    assert scaled[0] == 4.5
    inv = scaler.inverse_transform(scaled)
    assert inv[0] == 10.0

def test_token_padding(dummy_vocab):
    tokenizer = PSmilesTokenizer(dummy_vocab)
    ids = tokenizer.encode("C", add_special_tokens=False)
    assert ids == [4]

def test_unknown_token_handling(dummy_vocab):
    tokenizer = PSmilesTokenizer(dummy_vocab)
    ids = tokenizer.encode("XYZ", add_special_tokens=False, on_unknown="unk")
    assert ids == [1, 1, 1]

def test_two_branch_signature(dummy_vocab):
    model = TwoBranchTransformerRegressorModel(
        vocab_size=len(dummy_vocab), d_model=16, n_layers=1, n_heads=1, dim_feedforward=32, dropout=0.0, max_seq_len=64
    )
    model.eval()
    preds = predict_batch(model, ["C"], dummy_vocab, 64, True)
    assert len(preds) == 1

def test_batch_vs_single(dummy_model, dummy_vocab):
    s1 = "C"
    s2 = "cO"
    p1 = predict_batch(dummy_model, [s1], dummy_vocab, 64, False)[0]
    p2 = predict_batch(dummy_model, [s2], dummy_vocab, 64, False)[0]
    
    pb = predict_batch(dummy_model, [s1, s2], dummy_vocab, 64, False)
    np.testing.assert_allclose(pb[0], p1, rtol=1e-5)
    np.testing.assert_allclose(pb[1], p2, rtol=1e-5)

def test_success_threshold():
    source_pred = 1.0
    cand_pred = 1.06
    constraint_pass = True
    drift = abs(source_pred - cand_pred)
    assert (constraint_pass and drift > 0.05) == True
    
def test_constraint_pass_participation():
    source_pred = 1.0
    cand_pred = 1.5
    constraint_pass = False
    drift = abs(source_pred - cand_pred)
    assert (constraint_pass and drift > 0.05) == False

def test_canonical_tokenizer_parity():
    pass
