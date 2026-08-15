"""Training loop for Experiment 1 (clean baseline). PENDING(torch + dataset).

Requirements already settled for when this is written:
  - seed the loop explicitly (the sibling project did not, so its checkpoint was
    not reproducible despite its README claiming determinism)
  - save checkpoint, tokenizer vocabulary, split artifact, config and metrics
    together, so the baseline is reproducible from disk alone
  - record package versions and hardware in the run metadata
  - the TEST SET STAYS SEALED: model selection uses validation only
"""

from __future__ import annotations

from ..utils.pending import PendingImplementation


import json
import logging
import random
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..data.tokenizer import tokenize
from ..data.scaler import TargetScaler
from ..models.transformer import TransformerRegressorModel
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class PolymerDataset(Dataset):
    def __init__(self, data, vocab, max_len, scaler: TargetScaler):
        self.data = data
        self.char2idx = {c: i + 1 for i, c in enumerate(vocab)}
        self.max_len = max_len
        self.scaler = scaler
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = tokenize(row["psmiles"])
        ids = [self.char2idx.get(c, 0) for c in tokens]
        mask = [False] * len(ids)
        
        while len(ids) < self.max_len:
            ids.append(0)
            mask.append(True)
            
        ids = ids[:self.max_len]
        mask = mask[:self.max_len]
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target": torch.tensor(self.scaler.transform(row["target"]), dtype=torch.float32)
        }

def train(config_path: str = "configs/model.yaml", dataset_config_path: str = "configs/dataset.yaml", augmented_train_path: str = None, out_dir: str = "results/models/transformer_regressor", scaler_path: str = None, write_back_config: bool = True, seed: int = None):
    """Train the Tg regressor.

    seed:
        Overrides configs/model.yaml:training.seed. Phase 2C runs 5 independent
        seeds to estimate run-to-run variance, which every single-seed result up
        to Phase 2B lacked. Controls weight init, DataLoader shuffling and
        dropout masks.

    scaler_path:
        If given, LOAD this TargetScaler instead of fitting one on the training
        targets. This is the Phase 2B control: Phase 2A refit the scaler on the
        augmented set, shifting the normalization mean by ~10 K and confounding
        the clean-MAE comparison against Phase 1. Passing the Phase 1 scaler
        holds normalization statistics fixed so the only intended difference is
        adversarial augmentation.

    write_back_config:
        Phase 2A overwrote configs/model.yaml:baseline_metrics with the DEFENDED
        model's numbers. Pass False to leave the config untouched.
    """
    cfg = load_config(config_path)
    data_cfg = load_config(dataset_config_path)
    
    # Seeding. Covers weight init, DataLoader(shuffle=True) batch order and
    # dropout masks -- all three draw from torch's global RNG.
    seed = seed if seed is not None else cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Training seed: {seed}")
    
    proc_dir = Path(data_cfg["processed_dir"])
    df = pd.read_csv(proc_dir / "processed.csv")
    with open(proc_dir / "splits.json") as f:
        splits = json.load(f)
    with open(proc_dir / "vocab.json") as f:
        vocab = json.load(f)
        
    rep_col = data_cfg["representation_column"]
    tgt_col = data_cfg["target_column"]
    
    # We rename columns for convenience
    df = df.rename(columns={rep_col: "psmiles", tgt_col: "target"})
    
    max_len = cfg["architecture"]["max_seq_len"]
    
    
    if augmented_train_path:
        logger.info(f"Loading augmented training data from {augmented_train_path}")
        train_df = pd.read_csv(augmented_train_path)
        train_df = train_df.rename(columns={rep_col: "psmiles", tgt_col: "target"})
    else:
        train_df = df.iloc[splits["train"]]
        
    val_df = df.iloc[splits["val"]]
    test_df = df.iloc[splits["test"]]
    
    # Scaler: either load a fixed one (Phase 2B control) or fit on training targets.
    if scaler_path:
        scaler = TargetScaler.load(scaler_path)
        logger.info(
            f"CONTROL: loaded fixed scaler from {scaler_path} "
            f"(mean={scaler.mean:.6f}, std={scaler.std:.6f}) -- NOT refit on this training set"
        )
        fitted_mean = float(np.mean(train_df["target"].values))
        logger.info(
            f"  (for reference, refitting here would have given mean={fitted_mean:.6f}; "
            f"delta={fitted_mean - scaler.mean:+.6f} K)"
        )
    else:
        scaler = TargetScaler()
        scaler.fit(train_df["target"].values)
        logger.info(f"Fitted scaler on training targets (mean={scaler.mean:.6f}, std={scaler.std:.6f})")
    
    train_ds = PolymerDataset(train_df, vocab, max_len, scaler)
    val_ds = PolymerDataset(val_df, vocab, max_len, scaler)
    test_ds = PolymerDataset(test_df, vocab, max_len, scaler)
    
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    device = torch.device(cfg["device"])
    
    model = TransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=cfg["architecture"]["d_model"],
        n_layers=cfg["architecture"]["n_layers"],
        n_heads=cfg["architecture"]["n_heads"],
        dim_feedforward=cfg["architecture"]["dim_feedforward"],
        dropout=cfg["architecture"]["dropout"],
        max_seq_len=max_len,
        pooling=cfg["architecture"]["pooling"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["training"]["learning_rate"], 
        weight_decay=cfg["training"]["weight_decay"]
    )
    criterion = nn.MSELoss()
    
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    best_val_mae = float("inf")
    patience_counter = 0
    
    best_model_state = None
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            
            optimizer.zero_grad()
            preds = model(ids, padding_mask=mask)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ids)
            
        train_loss /= len(train_ds)
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["ids"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)
                preds = model(ids, padding_mask=mask)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
                
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # We compute MAE in the original Kelvin scale for logging
        val_preds_inv = scaler.inverse_transform(val_preds)
        val_targets_inv = scaler.inverse_transform(val_targets)
        val_mae = np.mean(np.abs(val_preds_inv - val_targets_inv))
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.4f} - Val MAE: {val_mae:.4f} K")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break
            
    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)
            preds = model(ids, padding_mask=mask)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
            
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    test_preds_inv = scaler.inverse_transform(test_preds)
    test_targets_inv = scaler.inverse_transform(test_targets)
    
    test_mae = np.mean(np.abs(test_preds_inv - test_targets_inv))
    test_rmse = np.sqrt(np.mean((test_preds_inv - test_targets_inv)**2))
    
    # R2 on original scale
    ss_tot = np.sum((test_targets_inv - np.mean(test_targets_inv))**2)
    ss_res = np.sum((test_targets_inv - test_preds_inv)**2)
    test_r2 = 1 - (ss_res / ss_tot)
    
    logger.info(f"Test Evaluation - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
    logger.info(f"Test Predictions (Sample): {test_preds_inv[:5]}")
    logger.info(f"Test Targets (Sample): {test_targets_inv[:5]}")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(best_model_state, out_dir / "model.pt")
    scaler.save(out_dir / "scaler.json")
    
    # Persist this run's own metrics next to its checkpoint, so a later run can
    # never overwrite an earlier run's recorded numbers.
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "test_r2": float(test_r2),
                "best_val_mae": float(best_val_mae),
                "scaler": {"mean": scaler.mean, "std": scaler.std},
                "scaler_source": scaler_path or "fitted_on_this_training_set",
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "seed": seed,
            },
            f,
            indent=2,
        )

    if write_back_config:
        cfg["baseline_metrics"] = {
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "r2": float(test_r2)
        }
        with open("configs/model.yaml", "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    logger.info(f"Saved checkpoint and metrics to {out_dir}")
    return model
