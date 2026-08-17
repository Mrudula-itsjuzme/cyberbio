import json
import yaml
import pandas as pd
from pathlib import Path
from rdkit import Chem
import logging

logging.basicConfig(level=logging.INFO)

def is_valid_smiles(smiles: str) -> bool:
    try:
        # Many PSMILES use [*] for attachments
        m = Chem.MolFromSmiles(smiles)
        return m is not None
    except Exception:
        return False

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("configs/dataset.yaml")
    interim_path = Path("data/interim/cleaned.csv")
    proc_dir = Path(cfg["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(interim_path)
    
    logging.info("Validating with RDKit...")
    valid_mask = df["original_representation"].apply(is_valid_smiles)
    df = df[valid_mask].copy()
    logging.info(f"Remaining after RDKit validation: {len(df)}")
    
    counts = df["original_representation"].value_counts()
    duplicates = counts[counts > 1].index
    if cfg.get("duplicate_policy") == "drop_all":
        df = df[~df["original_representation"].isin(duplicates)].copy()
    else:
        df = df.drop_duplicates(subset=["original_representation"], keep="first")
    logging.info(f"Remaining after deduplication: {len(df)}")
    
    split_cfg = cfg.get("split", {})
    train_frac = split_cfg.get("train_frac", 0.7)
    val_frac = split_cfg.get("val_frac", 0.15)
    test_frac = split_cfg.get("test_frac", 0.15)
    seed = split_cfg.get("seed", 42)
    
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    train_idx = int(len(df) * train_frac)
    val_idx = train_idx + int(len(df) * val_frac)
    
    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    combined = pd.concat([train_df, val_df, test_df])
    combined.to_csv(proc_dir / "processed.csv", index=False)
    
    splits = {
        "train": list(range(0, len(train_df))),
        "val": list(range(len(train_df), len(train_df) + len(val_df))),
        "test": list(range(len(train_df) + len(val_df), len(combined))),
        "test_sealed": split_cfg.get("test_sealed", True)
    }
    
    with open(proc_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
        
    logging.info(f"Split done. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Also build vocab
    from materials_adv.data.tokenizer import tokenize
    vocab = set()
    for _, row in train_df.iterrows():
        vocab.update(tokenize(row["original_representation"]))
    
    with open(proc_dir / "vocab.json", "w") as f:
        json.dump(sorted(list(vocab)), f)
        
if __name__ == "__main__":
    main()
