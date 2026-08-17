import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from materials_adv.attacks.generator import AttackGenerator
from materials_adv.attacks.probabilistic import ProbabilisticMCMCAttack
from materials_adv.utils.config import load_config
import torch
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.data.scaler import TargetScaler
from materials_adv.models.regression import TransformerRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading configs...")
    dataset_cfg = load_config("configs/dataset.yaml")
    model_cfg = load_config("configs/model.yaml")
    
    proc_dir = Path(dataset_cfg["processed_dir"])
    with open(proc_dir / "vocab.json") as f:
        vocab = json.load(f)
    with open(proc_dir / "splits.json") as f:
        splits = json.load(f)
        
    df = pd.read_csv(proc_dir / "processed.csv")
    rep_col = dataset_cfg["representation_column"]
    
    train_indices = splits["train"]
    train_samples = [(f"train_{idx}", df.iloc[idx][rep_col]) for idx in train_indices]
    
    logger.info("Loading residualized model for MCMC scoring...")
    device = torch.device(model_cfg["device"])
    base_model = TransformerRegressorModel(
        vocab_size=len(vocab), d_model=model_cfg["architecture"]["d_model"],
        n_layers=model_cfg["architecture"]["n_layers"], n_heads=model_cfg["architecture"]["n_heads"],
        dim_feedforward=model_cfg["architecture"]["dim_feedforward"], dropout=model_cfg["architecture"]["dropout"],
        max_seq_len=model_cfg["architecture"]["max_seq_len"], pooling=model_cfg["architecture"]["pooling"]
    ).to(device)
    
    model_dir = Path("results/models/transformer_regressor_residualized")
    base_model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location=device))
    base_model.eval()
    
    scaler = TargetScaler.load(model_dir / "scaler.json")
    predictor = TransformerRegressor(model=base_model, vocab=vocab, scaler=scaler, target_units="K")
    
    rng = np.random.default_rng(42)
    attack_mcmc = ProbabilisticMCMCAttack(
        rng=rng, predictor=predictor, allowed_tokens=vocab, steps=50, temperature=10.0,
        protect_attachments=True, protect_ring_closures=True, protect_branches=True
    )
    
    generator = AttackGenerator(attacks=[attack_mcmc], predictor=predictor, seed=42)
    
    logger.info(f"Generating MCMC attacks on {len(train_samples)} training samples (n=1)...")
    records = generator.run(train_samples, n_variants=1)
    
    valid_attacks = [r for r in records if r.validity_status == "valid"]
    
    # Build dataframe
    tgt_col = dataset_cfg["target_column"]
    target_map = {df.iloc[idx][rep_col]: df.iloc[idx][tgt_col] for idx in train_indices}
    
    new_rows = []
    for r in valid_attacks:
        orig_psmiles = r.original_psmiles
        if orig_psmiles in target_map:
            new_rows.append({
                rep_col: r.adversarial_psmiles,
                tgt_col: target_map[orig_psmiles],
                "dataset": "mcmc_augmented"
            })
            
    train_df = df.iloc[train_indices].copy()
    train_df["dataset"] = "clean"
    aug_df = pd.DataFrame(new_rows)
    combined = pd.concat([train_df, aug_df], ignore_index=True)
    combined.to_csv("data/processed/train_mcmc_augmented.csv", index=False)
    logger.info(f"Created augmented dataset with {len(new_rows)} MCMC examples. Total size: {len(combined)}")

if __name__ == "__main__":
    main()
