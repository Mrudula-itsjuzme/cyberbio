import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from materials_adv.attacks.generator import AttackGenerator
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_augmented_data():
    data_cfg = load_config("configs/dataset.yaml")
    attack_cfg = load_config("configs/attack.yaml")
    proc_dir = Path(data_cfg["processed_dir"])
    
    df = pd.read_csv(proc_dir / "processed.csv")
    with open(proc_dir / "splits.json") as f:
        splits = json.load(f)
    with open(proc_dir / "vocab.json") as f:
        vocab = json.load(f)
        
    train_indices = splits["train"]
    train_df = df.iloc[train_indices].copy()
    
    rng = np.random.default_rng(12345) # Seed for training augmentation
    
    attack_sub = SubstitutionAttack(
        rng=rng, allowed_tokens=vocab, n_edits=1, role_preserving=True,
        protect_attachments=True, protect_ring_closures=True, protect_branches=True
    )
    attack_ins = InsertionAttack(
        rng=rng, allowed_tokens=vocab, n_edits=1,
        protect_attachments=True, protect_ring_closures=True, protect_branches=True
    )
    attack_del = DeletionAttack(
        rng=rng, n_edits=1,
        protect_attachments=True, protect_ring_closures=True, protect_branches=True
    )
    attack_arr = RearrangementAttack(
        rng=rng, window_size=3,
        protect_attachments=True, protect_ring_closures=True, protect_branches=True
    )
    
    generator = AttackGenerator(
        attacks=[attack_sub, attack_ins, attack_del, attack_arr],
        predictor=None, # No predictor, we just want valid strings
        seed=12345,
        check_plausibility=True
    )
    
    augmented_rows = []
    total_train = len(train_df)
    
    rep_col = data_cfg["representation_column"]
    tgt_col = data_cfg["target_column"]
    
    logger.info(f"Generating adversarial training examples for {total_train} samples...")
    for i, (_, row) in enumerate(train_df.iterrows()):
        psmiles = row[rep_col]
        target = row[tgt_col]
        
        # 3 variants per attack family = max 12 candidates
        records = generator.run_sample(f"train_{i}", psmiles, n_variants=3)
        
        valid_records = [r for r in records if r.validity_status == "valid"]
        
        for r in valid_records:
            aug_row = row.copy()
            aug_row[rep_col] = r.adversarial_psmiles
            aug_row["is_adv"] = True
            aug_row["attack_type"] = r.attack_type
            augmented_rows.append(aug_row)
            
        if (i+1) % 50 == 0:
            logger.info(f"Processed {i+1}/{total_train}...")
            
    logger.info(f"Generated {len(augmented_rows)} valid adversarial examples.")
    
    aug_df = pd.DataFrame(augmented_rows)
    
    # We mark clean rows as not adv
    train_df["is_adv"] = False
    train_df["attack_type"] = "clean"
    
    # Combine
    combined_train_df = pd.concat([train_df, aug_df], ignore_index=True)
    
    out_path = proc_dir / "train_aug.csv"
    combined_train_df.to_csv(out_path, index=False)
    logger.info(f"Saved augmented training set ({len(combined_train_df)} rows) to {out_path}")

if __name__ == "__main__":
    generate_augmented_data()
