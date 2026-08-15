import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from materials_adv.attacks.generator import AttackGenerator
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config
from materials_adv.data.tokenizer import tokenize

logging.basicConfig(level=logging.INFO)

def run_smoke_test():
    dataset_cfg = load_config("configs/dataset.yaml")
    model_cfg = load_config("configs/model.yaml")
    attack_cfg = load_config("configs/attack.yaml")
    
    proc_dir = Path(dataset_cfg["processed_dir"])
    with open(proc_dir / "vocab.json") as f:
        vocab = json.load(f)
        
    df = pd.read_csv(proc_dir / "processed.csv")
    rep_col = dataset_cfg["representation_column"]
    
    # Pick 2 samples from the validation set for the smoke test
    with open(proc_dir / "splits.json") as f:
        splits = json.load(f)
    sample_indices = splits["val"][:2]
    
    device = torch.device("cpu")
    base_model = TransformerRegressorModel(
        vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4,
        dim_feedforward=128, dropout=0.1, max_seq_len=256, pooling="mean"
    )
    model_dir = Path("results/models/transformer_regressor")
    base_model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location=device))
    base_model.eval()
    scaler = TargetScaler.load(model_dir / "scaler.json")
    
    predictor = TransformerRegressor(base_model, vocab, scaler, dataset_cfg["target_units"])
    
    rng = np.random.default_rng(42)
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
        predictor=predictor, seed=42, check_plausibility=True
    )
    
    print("\n" + "="*50)
    print("SMOKE TEST: PHASE 1E ATTACKS")
    print("="*50)
    
    for idx in sample_indices:
        original_psmiles = df.iloc[idx][rep_col]
        print(f"\n[Sample {idx}] Original PSMILES: {original_psmiles}")
        original_tokens = tokenize(original_psmiles)
        print(f"Original Tokens: {original_tokens}")
        
        # n_variants=3 per attack
        records = generator.run_sample(f"sample_{idx}", original_psmiles, n_variants=3)
        
        for i, r in enumerate(records):
            print(f"\n  Variant {i+1} [{r.attack_type.upper()}]:")
            print(f"    Adversarial PSMILES: {r.adversarial_psmiles}")
            print(f"    Adv Tokens:          {tokenize(r.adversarial_psmiles)}")
            
            if r.attack_type == "substitution":
                adv_tokens = tokenize(r.adversarial_psmiles)
                diffs = [(pos, orig, adv) for pos, (orig, adv) in enumerate(zip(original_tokens, adv_tokens)) if orig != adv]
                for pos, orig, adv in diffs:
                    print(f"    Edit: Substitution: {orig} -> {adv} at position {pos}")
            elif r.attack_type == "insertion":
                print(f"    Edit: Insertion of {r.number_of_changes} tokens")
            elif r.attack_type == "deletion":
                print(f"    Edit: Deletion of {r.number_of_changes} tokens")
            elif r.attack_type == "rearrangement":
                print(f"    Edit: Rearrangement of {r.number_of_changes} tokens")
            
            print(f"    Representation Valid: {r.validity_status == 'valid'}")
            print(f"    Rejection Reasons:   {r.rejection_reasons}")
            print(f"    Original Pred:       {r.original_prediction:.2f} K")
            print(f"    Adversarial Pred:    {r.adversarial_prediction:.2f} K")
            print(f"    Prediction Drift:    {abs(r.prediction_drift):.2f} K")

if __name__ == "__main__":
    run_smoke_test()
