import pytest
import torch
import pandas as pd
import json
from pathlib import Path
from src.generative.model import DenoisingSeqAE
from src.generative.dataset import DenoisingDataset

ROOT = Path(__file__).resolve().parent.parent

def test_denoising_seq_ae_init():
    model = DenoisingSeqAE(vocab_size=10, pad_idx=0, d_embed=16, d_hidden=32, n_layers=1, dropout=0.0)
    assert model is not None
    
def test_denoising_seq_ae_forward():
    model = DenoisingSeqAE(vocab_size=10, pad_idx=0, d_embed=16, d_hidden=32, n_layers=1, dropout=0.0)
    x = torch.zeros((2, 5), dtype=torch.long)
    ti = torch.zeros((2, 5), dtype=torch.long)
    logits = model(x, ti)
    assert logits.shape == (2, 5, 11)
    
def test_denoising_seq_ae_generate():
    model = DenoisingSeqAE(vocab_size=10, pad_idx=0, d_embed=16, d_hidden=32, n_layers=1, dropout=0.0)
    x = torch.zeros((1, 5), dtype=torch.long)
    gen = model.generate(x, max_len=5, greedy=True)
    assert len(gen) <= 5

def test_denoising_dataset():
    df = pd.DataFrame({"sequence": ["ATCG", "GCTA"]})
    df.to_csv("/tmp/test_dataset.csv", index=False)
    vocab = {"A":1, "T":2, "C":3, "G":4}
    dataset = DenoisingDataset("/tmp/test_dataset.csv", vocab, max_len=4, noise_rate=0.0, column="sequence", pad_idx=0)
    x, ti, to = dataset[0]
    assert x.shape == (4,)
    
def test_checkpoints_exist():
    assert (ROOT / "results/generative/checkpoints/bio_cyber/checkpoint.pt").exists()
    assert (ROOT / "results/generative/checkpoints/materials/checkpoint.pt").exists()

def test_manifests_exist():
    assert (ROOT / "results/generative/bio_cyber/training_manifest.json").exists()
    assert (ROOT / "results/generative/materials/training_manifest.json").exists()

def test_results_csv_exists():
    assert (ROOT / "results/generative/bio_cyber/aggregated_results.csv").exists()
    assert (ROOT / "results/generative/materials/aggregated_results.csv").exists()

def test_cross_domain_extended():
    df = pd.read_csv(ROOT / "results/cross_domain/attack_strategy_comparison.csv")
    assert "Generative" in df["attack_condition"].values

def test_quality_summary_exists():
    assert (ROOT / "results/generative/generator_quality_summary.json").exists()
    
def test_figures_exist():
    assert (ROOT / "results/generative/generator_training_loss.png").exists()
    assert (ROOT / "results/cross_domain/cross_domain_efficiency.png").exists()
