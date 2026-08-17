import os
import yaml
import pandas as pd
from pathlib import Path

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config("configs/dataset.yaml")
    
    raw_dir = Path(cfg["raw_dir"])
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = raw_dir / cfg["file"]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Missing raw data file: {file_path}")
        
    df = pd.read_csv(file_path)
    
    rep_col = cfg["representation_column"]
    tgt_col = cfg["target_column"]
    tgt_prop = cfg.get("target_property", "Tg")
    tgt_units = cfg.get("target_units", "unknown")
    
    # Filter missing values
    df = df.dropna(subset=[rep_col, tgt_col])
    
    # Map to standardized schema
    # Schema: polymer_id, original_representation, property_name, property_value, units, source_dataset
    records = []
    for idx, row in df.iterrows():
        records.append({
            "polymer_id": f"poly_{idx}",
            "original_representation": str(row[rep_col]),
            "property_name": tgt_prop,
            "property_value": float(row[tgt_col]),
            "units": tgt_units,
            "source_dataset": cfg["name"]
        })
        
    cleaned_df = pd.DataFrame(records)
    out_path = interim_dir / "cleaned.csv"
    cleaned_df.to_csv(out_path, index=False)
    
    print(f"Preprocessed {len(cleaned_df)} records for property '{tgt_prop}'. Saved to {out_path}.")

if __name__ == "__main__":
    main()
