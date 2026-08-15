import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_metric(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_tot)

from materials_adv.data.tokenizer import tokenize
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanity")

def main():
    print("="*50)
    print("1. VERIFY ACTUAL DATA")
    print("="*50)
    
    cfg = load_config("configs/dataset.yaml")
    raw_path = Path(cfg["raw_dir"]) / cfg["file"]
    raw_df = pd.read_csv(raw_path)
    
    rep_col = cfg["representation_column"]
    tgt_col = cfg["target_column"]
    
    print(f"Total raw rows: {len(raw_df)}")
    
    valid_df = raw_df.dropna(subset=[rep_col, tgt_col]).copy()
    print(f"Usable Tg rows (non-null PSMILES & Tg): {len(valid_df)}")
    
    print(f"Unique PSMILES in usable: {valid_df[rep_col].nunique()}")
    
    # We canonicalized in preprocessing
    from materials_adv.data.preprocessing import canonicalize
    valid_df["canonical"] = valid_df[rep_col].apply(canonicalize)
    valid_df = valid_df.dropna(subset=["canonical"])
    print(f"Unique canonical representations: {valid_df['canonical'].nunique()}")
    
    # Conflict check
    groups = valid_df.groupby("canonical")[tgt_col].apply(lambda x: x.max() - x.min())
    conflicts = groups[groups > 0]
    print(f"Number of conflicting duplicates: {len(conflicts)}")
    
    proc_df = pd.read_csv(Path(cfg["processed_dir"]) / "processed.csv")
    print(f"Number removed by conflict policy (drop_all): {len(valid_df) - len(proc_df)}")
    print(f"Number of final samples: {len(proc_df)}")
    
    with open(Path(cfg["processed_dir"]) / "splits.json") as f:
        splits = json.load(f)
        
    print(f"Train samples (scaffold groups): {len(splits['train'])}")
    print(f"Validation samples (scaffold groups): {len(splits['val'])}")
    print(f"Test samples (scaffold groups): {len(splits['test'])}")
    
    print("\n" + "="*50)
    print("2. CHECK THE TARGET SCALE")
    print("="*50)
    print(f"Tg minimum: {proc_df[tgt_col].min()}")
    print(f"Tg maximum: {proc_df[tgt_col].max()}")
    print(f"Tg mean: {proc_df[tgt_col].mean():.2f}")
    print(f"Tg median: {proc_df[tgt_col].median():.2f}")
    print(f"Tg std: {proc_df[tgt_col].std():.2f}")
    print(f"Tg units: {cfg['target_units']}")
    
    print("\n" + "="*50)
    print("3. COMPARE AGAINST SIMPLE BASELINES")
    print("="*50)
    
    train_targets = proc_df.iloc[splits["train"]][tgt_col].values
    test_targets = proc_df.iloc[splits["test"]][tgt_col].values
    
    mean_pred = np.full_like(test_targets, train_targets.mean())
    median_pred = np.full_like(test_targets, np.median(train_targets))
    
    for name, pred in [("Mean Predictor", mean_pred), ("Median Predictor", median_pred)]:
        mae = mae_score(test_targets, pred)
        rmse = rmse_score(test_targets, pred)
        r2 = r2_metric(test_targets, pred)
        print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
        
    print("\n" + "="*50)
    print("4 & 5. MODEL CAPACITY & PREDICTION EXAMPLES")
    print("="*50)
    
    model_cfg = load_config("configs/model.yaml")
    with open(Path(cfg["processed_dir"]) / "vocab.json") as f:
        vocab = json.load(f)
        
    print(f"Vocab size: {len(vocab)}")
    print(f"Model capacity: {model_cfg['architecture']}")
    
    model = TransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=model_cfg["architecture"]["d_model"],
        n_layers=model_cfg["architecture"]["n_layers"],
        n_heads=model_cfg["architecture"]["n_heads"],
        dim_feedforward=model_cfg["architecture"]["dim_feedforward"],
        dropout=model_cfg["architecture"]["dropout"],
        max_seq_len=model_cfg["architecture"]["max_seq_len"],
        pooling=model_cfg["architecture"]["pooling"]
    )
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    try:
        model.load_state_dict(torch.load("results/models/transformer_regressor/model.pt", weights_only=True))
        model.eval()
        
        from materials_adv.data.scaler import TargetScaler
        scaler = TargetScaler.load("results/models/transformer_regressor/scaler.json")
        
        char2idx = {c: i + 1 for i, c in enumerate(vocab)}
        max_len = model_cfg["architecture"]["max_seq_len"]
        
        print("\nTest Set Prediction Examples:")
        print(f"{'True Tg':<10} | {'Pred Tg':<10} | {'Abs Error':<10}")
        print("-" * 35)
        
        preds = []
        for idx in splits["test"][:5]:
            row = proc_df.iloc[idx]
            true_tg = row[tgt_col]
            
            tokens = tokenize(row[rep_col])
            ids = [char2idx.get(c, 0) for c in tokens]
            mask = [False] * len(ids)
            while len(ids) < max_len:
                ids.append(0)
                mask.append(True)
            ids = ids[:max_len]
            mask = mask[:max_len]
            
            with torch.no_grad():
                pred_tg_norm = model(torch.tensor([ids]), padding_mask=torch.tensor([mask])).item()
            pred_tg = scaler.inverse_transform(pred_tg_norm)
            preds.append(pred_tg)
            print(f"{true_tg:<10.2f} | {pred_tg:<10.2f} | {abs(true_tg - pred_tg):<10.2f}")
            
        print(f"\nModel prediction range on sample: {min(preds):.2f} to {max(preds):.2f}")
    except Exception as e:
        print("Could not load/run model:", e)
        
    print("\n" + "="*50)
    print("7. CHECK TOKENIZER")
    print("="*50)
    
    from materials_adv.data.tokenizer import detokenize
    mismatches = 0
    max_len_found = 0
    unk_count = 0
    for sm in proc_df[rep_col]:
        toks = tokenize(sm)
        max_len_found = max(max_len_found, len(toks))
        unk_count += toks.count("<unk>")
        recon = detokenize(toks)
        if recon != sm:
            mismatches += 1
            
    print(f"Mismatches in roundtrip: {mismatches} / {len(proc_df)}")
    print(f"Max seq len found: {max_len_found}")
    print(f"Unknown tokens: {unk_count}")

if __name__ == "__main__":
    main()
