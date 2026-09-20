#!/usr/bin/env python3
"""Generate the Contextual Thompson source cohort and fixed-arm training data."""

import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

from materials_adv.utils.config import load_config

def main():
    out_dir = Path("results/contextual_thompson")
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "context_source_split.json"
    
    dataset_config = load_config("dataset")
    processed = Path(dataset_config["processed_dir"])
    df = pd.read_csv(processed / "processed.csv")
    canonical_splits = json.loads((processed / "splits.json").read_text())
    
    # Generate dataset hash based on canonical splits and lengths to document version
    split_str = json.dumps(canonical_splits, sort_keys=True)
    dataset_hash = hashlib.md5(split_str.encode("utf-8")).hexdigest()
    
    seed = 20260920
    rng = np.random.default_rng(seed)
    
    # Use canonical 'train' for context train, 'val' for context val, 'test' for context test.
    # To keep experiments tractable, sample up to a certain N per split if they are huge.
    # The user asked for minimum 50, preferred 100+.
    # We will pick 150 train, 30 val, 50 test from the canonical splits to form the context cohort.
    
    def sample_split(indices, n, rng_instance):
        indices = np.array(indices)
        if len(indices) > n:
            return rng_instance.choice(indices, size=n, replace=False).tolist()
        return indices.tolist()
    
    train_ids = sample_split(canonical_splits.get("train", []), 150, rng)
    val_ids   = sample_split(canonical_splits.get("val", []), 30, rng)
    test_ids  = sample_split(canonical_splits.get("test", []), 50, rng)
    
    # Enforce zero overlap
    assert len(set(train_ids).intersection(val_ids)) == 0, "Overlap train/val!"
    assert len(set(train_ids).intersection(test_ids)) == 0, "Overlap train/test!"
    assert len(set(val_ids).intersection(test_ids)) == 0, "Overlap val/test!"
    
    # Store IDs as row identifiers
    cohort_metadata = {
        "train_ids": [f"row_{idx}" for idx in train_ids],
        "validation_ids": [f"row_{idx}" for idx in val_ids],
        "test_ids": [f"row_{idx}" for idx in test_ids],
        "split_strategy": "canonical_subset",
        "seed": seed,
        "dataset_hash": dataset_hash
    }
    
    with open(split_path, "w") as f:
        json.dump(cohort_metadata, f, indent=2)
        
    print(f"Context split saved to {split_path}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

if __name__ == "__main__":
    main()
