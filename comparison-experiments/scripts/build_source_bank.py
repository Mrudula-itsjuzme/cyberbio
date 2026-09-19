import json
import random
import hashlib
from pathlib import Path
import pandas as pd

def main():
    repo_root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    splits_path = repo_root / "data/processed/splits.json"
    data_path = repo_root / "data/processed/processed.csv"
    
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    test_ids = splits.get("test", [])
    if not test_ids:
        print("No test IDs found.")
        return
        
    print(f"Total test set size: {len(test_ids)}")
    
    df = pd.read_csv(data_path)
    # Ensure ID columns
    # We don't know the exact column name, but let's assume it's index or 'id'
    # Actually, splits might be integer indices or actual string IDs.
    print(f"First few test_ids: {test_ids[:5]}")
    
    random.seed(42)
    sample_size = min(100, len(test_ids))
    sampled_ids = random.sample(test_ids, sample_size)
    sampled_ids.sort() # for reproducibility
    
    # Hash of sampled IDs
    ids_str = ",".join(str(x) for x in sampled_ids)
    bank_hash = hashlib.sha256(ids_str.encode("utf-8")).hexdigest()
    
    output_dir = Path("/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "frozen_source_ids.json"
    
    frozen_data = {
        "description": "Frozen source bank for exploratory comparison framework",
        "seed": 42,
        "sample_size": sample_size,
        "bank_hash": bank_hash,
        "source_ids": sampled_ids
    }
    
    with open(output_path, "w") as f:
        json.dump(frozen_data, f, indent=4)
        
    print(f"Saved {sample_size} IDs to {output_path}")
    print(f"Bank Hash: {bank_hash}")
    
if __name__ == "__main__":
    main()
