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
from materials_adv.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.probabilistic import ProbabilisticMCMCAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="results/models/transformer_regressor")
    parser.add_argument("--out-file", type=str, default="phase1e_results.jsonl")
    parser.add_argument(
        "--attack-seed",
        type=int,
        default=None,
        help=(
            "Override configs/attack.yaml seed. Phase 2B uses a FRESH seed so the "
            "candidate set differs from Phase 1E, while holding it IDENTICAL across "
            "the models being compared -- the attacks are generated from the sample "
            "PSMILES, not from model gradients, so the same seed yields the same "
            "candidate strings for every model. That makes the comparison paired."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the drift threshold (default: baseline_metrics.mae from model.yaml).",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to attack.")
    args = parser.parse_args()

    logger.info("Loading configs...")
    dataset_cfg = load_config("configs/dataset.yaml")
    model_cfg = load_config("configs/model.yaml")
    attack_cfg = load_config("configs/attack.yaml")
    
    proc_dir = Path(dataset_cfg["processed_dir"])
    with open(proc_dir / "vocab.json") as f:
        vocab = json.load(f)
        
    with open(proc_dir / "splits.json") as f:
        splits = json.load(f)
        
    df = pd.read_csv(proc_dir / "processed.csv")
    rep_col = "original_representation"
    
    target_indices = splits[args.split]
    val_samples = [(f"sample_{idx}", df.iloc[idx][rep_col]) for idx in target_indices]
    
    logger.info("Loading baseline model and scaler...")
    device = torch.device(model_cfg["device"])
    base_model = TransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=model_cfg["architecture"]["d_model"],
        n_layers=model_cfg["architecture"]["n_layers"],
        n_heads=model_cfg["architecture"]["n_heads"],
        dim_feedforward=model_cfg["architecture"]["dim_feedforward"],
        dropout=model_cfg["architecture"]["dropout"],
        max_seq_len=model_cfg["architecture"]["max_seq_len"],
        pooling=model_cfg["architecture"]["pooling"]
    ).to(device)
    
    model_dir = Path(args.model_dir)
    base_model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location=device))
    base_model.eval()
    
    scaler = TargetScaler.load(model_dir / "scaler.json")
    
    predictor = TransformerRegressor(
        model=base_model,
        vocab=vocab,
        scaler=scaler,
        target_units=dataset_cfg["target_units"]
    )
    
    # Configure Substitution Attack
    attack_seed = args.attack_seed if args.attack_seed is not None else attack_cfg["seed"]
    logger.info(f"Attack RNG seed: {attack_seed}")
    rng = np.random.default_rng(attack_seed)
    
    sub_cfg = attack_cfg["attacks"]["substitution"]
    if not sub_cfg["enabled"]:
        logger.warning("Substitution attack is disabled in config. Enabling it for Phase 1D.")
        
    logger.info(f"Injecting vocabulary pool ({len(vocab)} tokens) into substitution attack.")
    attack_sub = SubstitutionAttack(
        rng=rng,
        allowed_tokens=vocab,
        attack_budget=sub_cfg["attack_budget"],
        role_preserving=sub_cfg["role_preserving"],
        protect_attachments=attack_cfg["protection"]["protect_attachments"],
        protect_ring_closures=attack_cfg["protection"]["protect_ring_closures"],
        protect_branches=attack_cfg["protection"]["protect_branches"]
    )
    
    attack_ins = InsertionAttack(
        rng=rng, allowed_tokens=vocab, attack_budget=attack_cfg['attacks']['insertion']['attack_budget'],
        protect_attachments=attack_cfg["protection"]["protect_attachments"],
        protect_ring_closures=attack_cfg["protection"]["protect_ring_closures"],
        protect_branches=attack_cfg["protection"]["protect_branches"]
    )
    
    attack_del = DeletionAttack(
        rng=rng, attack_budget=attack_cfg['attacks']['insertion']['attack_budget'],
        protect_attachments=attack_cfg["protection"]["protect_attachments"],
        protect_ring_closures=attack_cfg["protection"]["protect_ring_closures"],
        protect_branches=attack_cfg["protection"]["protect_branches"]
    )
    
    attack_arr = RearrangementAttack(
        rng=rng, window_size=3,
        protect_attachments=attack_cfg["protection"]["protect_attachments"],
        protect_ring_closures=attack_cfg["protection"]["protect_ring_closures"],
        protect_branches=attack_cfg["protection"]["protect_branches"]
    )
    
    attack_rand = SmilesRandomizationAttack(
        rng=rng,
        protect_attachments=attack_cfg["protection"]["protect_attachments"],
        protect_ring_closures=attack_cfg["protection"]["protect_ring_closures"],
        protect_branches=attack_cfg["protection"]["protect_branches"]
    )
    
    attack_mcmc = ProbabilisticMCMCAttack(
        rng=rng,
        predictor=predictor,
        allowed_tokens=vocab,
        steps=50,
        temperature=10.0,
        protect_attachments=attack_cfg["protection"]["protect_attachments"],
        protect_ring_closures=attack_cfg["protection"]["protect_ring_closures"],
        protect_branches=attack_cfg["protection"]["protect_branches"]
    )
    
    # Configure Generator
    # Set min_abs_drift dynamically from baseline
    # NOTE: configs/model.yaml:baseline_metrics was overwritten by the Phase 2A run
    # (it now holds the DEFENDED model's numbers). The threshold must stay pinned to
    # the Phase 1 baseline value for comparability across phases, so it is passed
    # explicitly rather than read from the config.
    if args.threshold is not None:
        baseline_mae = args.threshold
    else:
        baseline_mae = model_cfg.get("baseline_metrics", {}).get("mae", 52.0)
        logger.warning(
            "No --threshold given; reading from configs/model.yaml (=%.2f). "
            "This file was overwritten by Phase 2A -- pass --threshold 52.02 for "
            "comparability with Phase 1E.",
            baseline_mae,
        )
    logger.info(f"Using {baseline_mae:.2f} K as min_abs_drift threshold (informative).")
    
    generator = AttackGenerator(
        attacks=[attack_sub, attack_ins, attack_del, attack_arr, attack_rand, attack_mcmc],
        predictor=predictor,
        seed=attack_cfg["seed"],
        check_plausibility=attack_cfg["validation"]["check_plausibility"]
    )
    
    logger.info(f"Running attacks on {len(val_samples)} validation samples with {attack_cfg['n_variants']} variants each...")
    records = generator.run(val_samples, n_variants=attack_cfg["n_variants"])
    
    out_dir = Path(attack_cfg["output"]["records_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_file
    
    import dataclasses
    with open(out_path, "w") as f:
        for r in records:
            # We add an 'is_successful' flag based on drift and validity
            # For our record-keeping context
            success = (
                r.validity_status == "valid" and
                r.signed_prediction_drift is not None and
                abs(r.signed_prediction_drift) > baseline_mae
            )
            data = r.to_dict()
            data["is_successful"] = bool(success)
            f.write(json.dumps(data) + "\n")
            
    # Metrics
    valid_records = [r for r in records if r.validity_status == "valid"]
    successful_records = [r for r in records if (r.validity_status == "valid" and r.signed_prediction_drift is not None and abs(r.signed_prediction_drift) > baseline_mae)]
    
    logger.info(f"Total attacks generated: {len(records)}")
    logger.info(f"Valid candidates: {len(valid_records)} ({len(valid_records)/len(records):.1%})")
    logger.info(f"Successful attacks (> {baseline_mae:.2f} K drift): {len(successful_records)} ({len(successful_records)/len(records):.1%})")
    
    if valid_records:
        drifts = [abs(r.signed_prediction_drift) for r in valid_records if r.signed_prediction_drift is not None]
        logger.info(f"Average absolute drift (valid only): {np.mean(drifts):.2f} K")
        logger.info(f"Max absolute drift (valid only): {np.max(drifts):.2f} K")
        
    logger.info(f"Saved records to {out_path}")

if __name__ == "__main__":
    main()
