"""Before/After Adversarial Defense Experiment Script.

Workflow:
1. Load polymer property dataset (PSMILES -> Bandgap/Property value).
2. Train baseline Transformer regressor model on clean training set.
3. Attack baseline model with Probabilistic MCMC Attack Generator (with chemical plausibility constraints).
4. Measure baseline robustness metrics: Clean RMSE/MAE, Adv RMSE/MAE, Mean Absolute Drift, Attack Success Rate (ASR).
5. Perform closed-loop adversarial training (min-max loop) to produce defended model.
6. Attack defended model with Probabilistic MCMC Attack Generator.
7. Measure defended robustness metrics and compute before/after improvement ratios.
8. Output formatted summary table and persist results JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize, Vocabulary
from materials_adv.domain.chemistry.attacks.probabilistic import ProbabilisticMCMCAttack
from materials_adv.domain.chemistry.plausibility import ChemicalPlausibilityValidator
from materials_adv.evaluation.metrics import regression_metrics, robustness_metrics
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PolymerDataset(Dataset):
    def __init__(self, representations: list[str], targets: list[float], vocab_map: dict[str, int], max_len: int = 128):
        self.representations = representations
        self.targets = targets
        self.vocab_map = vocab_map
        self.max_len = max_len

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        rep = self.representations[idx]
        tokens = tokenize(rep)
        seq_len = min(len(tokens), self.max_len)
        
        token_ids = torch.zeros(self.max_len, dtype=torch.long)
        mask = torch.ones(self.max_len, dtype=torch.bool)
        
        for j in range(seq_len):
            token_ids[j] = self.vocab_map.get(tokens[j], 0)
            mask[j] = False
            
        return {
            "representation": rep,
            "token_ids": token_ids,
            "mask": mask,
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class ModelPredictorAdapter:
    """Predictor wrapper for evaluation and attack generator integration."""
    def __init__(self, model: nn.Module, vocab_map: dict[str, int], scaler: TargetScaler, device: str = "cpu", max_len: int = 128):
        self.model = model
        self.vocab_map = vocab_map
        self.scaler = scaler
        self.device = device
        self.max_len = max_len
        self.model.to(device)
        self.model.eval()

    def predict(self, representations: list[str]) -> list[float]:
        if not representations:
            return []
        
        preds = []
        with torch.no_grad():
            for rep in representations:
                tokens = tokenize(rep)
                seq_len = min(len(tokens), self.max_len)
                token_ids = torch.zeros((1, self.max_len), dtype=torch.long, device=self.device)
                mask = torch.ones((1, self.max_len), dtype=torch.bool, device=self.device)
                
                for j in range(seq_len):
                    token_ids[0, j] = self.vocab_map.get(tokens[j], 0)
                    mask[0, j] = False
                    
                scaled_pred = self.model(token_ids, padding_mask=mask).cpu().numpy().item()
                unscaled = float(self.scaler.inverse_transform(np.array([[scaled_pred]]))[0, 0])
                preds.append(unscaled)
        return preds


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 5,
    adv_generator: ProbabilisticMCMCAttack | None = None,
    adv_lambda: float = 0.5,
) -> list[float]:
    model.to(device)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            preds = model(token_ids, padding_mask=mask)
            clean_loss = criterion(preds, targets)
            
            total_loss = clean_loss
            
            # Closed-Loop Adversarial Training Step
            if adv_generator is not None and adv_lambda > 0.0:
                # Generate adversarial candidates for mini-batch
                reps = batch["representation"]
                adv_reps = []
                for rep in reps[:4]: # batch subset for efficiency
                    outcomes = adv_generator.generate(tokenize(rep), n_variants=1)
                    if outcomes:
                        adv_reps.append(outcomes[0].adversarial_representation)
                    else:
                        adv_reps.append(rep)
                
                if adv_reps:
                    # Tokenize adversarial batch
                    adv_ids = torch.zeros((len(adv_reps), train_loader.dataset.max_len), dtype=torch.long, device=device)
                    adv_mask = torch.ones((len(adv_reps), train_loader.dataset.max_len), dtype=torch.bool, device=device)
                    for i, r in enumerate(adv_reps):
                        toks = tokenize(r)
                        slen = min(len(toks), train_loader.dataset.max_len)
                        for j in range(slen):
                            adv_ids[i, j] = train_loader.dataset.vocab_map.get(toks[j], 0)
                            adv_mask[i, j] = False
                    
                    adv_preds = model(adv_ids, padding_mask=adv_mask)
                    adv_targets = targets[:len(adv_reps)]
                    adv_loss = criterion(adv_preds, adv_targets)
                    total_loss = (1.0 - adv_lambda) * clean_loss + adv_lambda * adv_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batches += 1
            
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return losses


def run_before_after_experiment(data_path: str = "data/processed/processed.csv", output_dir: str = "results"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    # Check representation & property column names
    rep_col = "original_representation" if "original_representation" in df.columns else "smiles"
    prop_col = "property_value" if "property_value" in df.columns else "bandgap"
    
    if rep_col not in df.columns or prop_col not in df.columns:
        # Fallback synthetic demonstration data if file format differs
        logger.info("Using dataset fallback format...")
        df = df.dropna().head(100)

    reps = df[rep_col].astype(str).tolist()[:100]
    vals = df[prop_col].astype(float).tolist()[:100]

    # Build vocab and scaler
    vocab_obj = Vocabulary.build(reps)
    vocab = vocab_obj.itos
    vocab_map = vocab_obj.stoi
    
    scaler = TargetScaler()
    scaler.fit(np.array(vals))
    scaled_vals = scaler.transform(np.array(vals)).flatten().tolist()

    # Split train/test
    split_idx = int(len(reps) * 0.8)
    train_reps, test_reps = reps[:split_idx], reps[split_idx:]
    train_vals, test_vals = scaled_vals[:split_idx], scaled_vals[split_idx:]
    test_true = vals[split_idx:]

    train_dataset = PolymerDataset(train_reps, train_vals, vocab_map)
    test_dataset = PolymerDataset(test_reps, test_vals, vocab_map)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 1. Train Baseline Model
    logger.info("--- Phase 1: Training Baseline Transformer Model ---")
    baseline_model = TwoBranchTransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    train_model(baseline_model, train_loader, optimizer, device=device, epochs=5)

    baseline_adapter = ModelPredictorAdapter(baseline_model, vocab_map, scaler, device=device)

    # 2. Attack Baseline Model
    logger.info("--- Phase 2: Attacking Baseline Model with Probabilistic MCMC Attack ---")
    rng = np.random.default_rng(42)
    mcmc_attack = ProbabilisticMCMCAttack(
        rng=rng,
        predictor=baseline_adapter,
        allowed_tokens=vocab,
        steps=20,
        temperature=5.0,
        min_tanimoto_similarity=0.5,
    )
    validator = ChemicalPlausibilityValidator(min_tanimoto_similarity=0.5)

    clean_preds_base = baseline_adapter.predict(test_reps)
    adv_reps_base = []
    valid_flags_base = []

    for r in test_reps:
        outcomes = mcmc_attack.generate(tokenize(r), n_variants=1)
        if outcomes:
            cand = outcomes[0].adversarial_representation
            is_valid, _ = validator.validate(r, cand)
            adv_reps_base.append(cand)
            valid_flags_base.append(is_valid)
        else:
            adv_reps_base.append(r)
            valid_flags_base.append(True)

    adv_preds_base = baseline_adapter.predict(adv_reps_base)
    base_robustness = robustness_metrics(test_true, clean_preds_base, adv_preds_base, valid_flags_base)
    logger.info(f"Baseline Metrics: {json.dumps(base_robustness, indent=2)}")

    # 3. Closed-Loop Adversarial Training (Defender)
    logger.info("--- Phase 3: Adversarial Defender Training (Closed-Loop Min-Max Optimization) ---")
    defended_model = TwoBranchTransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4)
    defended_optimizer = torch.optim.Adam(defended_model.parameters(), lr=1e-3)
    
    adv_trainer_generator = ProbabilisticMCMCAttack(
        rng=np.random.default_rng(123),
        predictor=baseline_adapter,
        allowed_tokens=vocab,
        steps=10,
        min_tanimoto_similarity=0.5,
    )
    
    train_model(defended_model, train_loader, defended_optimizer, device=device, epochs=5, adv_generator=adv_trainer_generator, adv_lambda=0.5)

    defended_adapter = ModelPredictorAdapter(defended_model, vocab_map, scaler, device=device)

    # 4. Attack Defended Model
    logger.info("--- Phase 4: Attacking Defended Model ---")
    mcmc_attack_def = ProbabilisticMCMCAttack(
        rng=np.random.default_rng(42),
        predictor=defended_adapter,
        allowed_tokens=vocab,
        steps=20,
        temperature=5.0,
        min_tanimoto_similarity=0.5,
    )

    clean_preds_def = defended_adapter.predict(test_reps)
    adv_reps_def = []
    valid_flags_def = []

    for r in test_reps:
        outcomes = mcmc_attack_def.generate(tokenize(r), n_variants=1)
        if outcomes:
            cand = outcomes[0].adversarial_representation
            is_valid, _ = validator.validate(r, cand)
            adv_reps_def.append(cand)
            valid_flags_def.append(is_valid)
        else:
            adv_reps_def.append(r)
            valid_flags_def.append(True)

    adv_preds_def = defended_adapter.predict(adv_reps_def)
    defended_robustness = robustness_metrics(test_true, clean_preds_def, adv_preds_def, valid_flags_def)
    logger.info(f"Defended Metrics: {json.dumps(defended_robustness, indent=2)}")

    # 5. Compute Comparative Summary
    drift_reduction = (base_robustness["mean_absolute_drift"] - defended_robustness["mean_absolute_drift"]) / max(base_robustness["mean_absolute_drift"], 1e-9)
    asr_reduction = (base_robustness["attack_success_rate_tau_0.5"] - defended_robustness["attack_success_rate_tau_0.5"]) / max(base_robustness["attack_success_rate_tau_0.5"], 1e-9)

    summary = {
        "baseline_model": base_robustness,
        "defended_model": defended_robustness,
        "improvements": {
            "drift_reduction_percentage": float(drift_reduction * 100.0),
            "asr_reduction_percentage": float(asr_reduction * 100.0),
        }
    }

    # Print Comparison Table
    print("\n==========================================================================================")
    print("                      BEFORE VS AFTER ADVERSARIAL DEFENSE COMPARISON                     ")
    print("==========================================================================================")
    print(f"{'Metric':<35} | {'Baseline Model':<20} | {'Defended Model':<20}")
    print("------------------------------------------------------------------------------------------")
    print(f"{'Clean RMSE (eV)':<35} | {base_robustness['clean_rmse']:<20.4f} | {defended_robustness['clean_rmse']:<20.4f}")
    print(f"{'Clean MAE (eV)':<35} | {base_robustness['clean_mae']:<20.4f} | {defended_robustness['clean_mae']:<20.4f}")
    print(f"{'Adversarial RMSE (eV)':<35} | {base_robustness['adv_rmse']:<20.4f} | {defended_robustness['adv_rmse']:<20.4f}")
    print(f"{'Mean Absolute Drift (eV)':<35} | {base_robustness['mean_absolute_drift']:<20.4f} | {defended_robustness['mean_absolute_drift']:<20.4f}")
    print(f"{'Max Absolute Drift (eV)':<35} | {base_robustness['max_absolute_drift']:<20.4f} | {defended_robustness['max_absolute_drift']:<20.4f}")
    print(f"{'Attack Success Rate (tau=0.5 eV)':<35} | {base_robustness['attack_success_rate_tau_0.5']:<20.4f} | {defended_robustness['attack_success_rate_tau_0.5']:<20.4f}")
    print(f"{'Validity Rate':<35} | {base_robustness['validity_rate']:<20.4f} | {defended_robustness['validity_rate']:<20.4f}")
    print("------------------------------------------------------------------------------------------")
    print(f"Drift Reduction: {drift_reduction*100.0:.2f}% | ASR (tau=0.5 eV) Reduction: {asr_reduction*100.0:.2f}%")
    print("==========================================================================================\n")

    res_file = out_dir / "before_after_defense_experiment_results.json"
    res_file.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved results to {res_file}")


if __name__ == "__main__":
    run_before_after_experiment()
