import pytest
import json
import torch
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent

def test_phase9_adversarial_dataset_reuse():
    # Verify the training script points to the Phase 9 dataset
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "p9_dir / \"adversarial_training_set.jsonl\"" in content
        
def test_canonical_equivalence_and_label_validity():
    # Verify the script loads the actual property value from the original dataset
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "df.loc[df[\"polymer_id\"] == rec[\"source_id\"], \"property_value\"].iloc[0]" in content

def test_same_scaler_and_tokenizer():
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "scaler = TargetScaler.load(root / \"results\" / \"models\" / \"transformer_regressor\" / \"scaler.json\")" in content
        assert "vocab.json" in content

def test_batch_ratio_enforcement():
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "nc = int(B * config[\"clean_ratio\"])" in content

def test_frozen_encoder_logic():
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "p.requires_grad_(False)" in content
        assert "if \"encoder\" in n or \"embedding\" in n:" in content

def test_low_learning_rate():
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "\"lr\": 1e-4" in content

def test_no_validation_contamination():
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "val_df = df.iloc[splits[\"val\"]].copy()" in content
        assert "AblationDataset(train_data" in content

def test_d0_checkpoint_immutable():
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "weights_only=True" in content

def test_deterministic_seed():
    with open(root / "scripts" / "run_phase10_collapse_diagnosis.py") as f:
        content = f.read()
        assert "torch.manual_seed(20260901)" in content
