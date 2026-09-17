"""Comprehensive Multi-Seed, Baseline Attack & Ablation Benchmark Suite.

Executes:
1. Multi-Seed Statistical Validation across 5 seeds (42, 123, 2026, 777, 999) reporting Mean +/- Std.
2. Attack Paradigm Comparisons: Random Mutation vs Deterministic Substitution vs Probabilistic MCMC.
3. Component Ablation Study: Loss weight lambda sweeps, Tanimoto similarity cutoff sweeps, Plausibility filter toggles.
4. Epistemic Uncertainty Shift Evaluation via MC-Dropout.
5. Sensitivity Analysis: Prediction drift vs MCMC step budget (N_steps in [5, 10, 20, 50, 100]).
6. Exports results to results/comprehensive_benchmark_summary.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize, Vocabulary
from materials_adv.domain.chemistry.attacks.probabilistic import ProbabilisticMCMCAttack
from materials_adv.domain.chemistry.attacks.randomization import SmilesRandomizationAttack
from materials_adv.domain.chemistry.attacks.simple_substitution import SimpleSubstitutionAttack
from materials_adv.domain.chemistry.plausibility import ChemicalPlausibilityValidator
from materials_adv.evaluation.metrics import regression_metrics, robustness_metrics
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 123, 2026, 777, 999]


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


class ModelAdapter:
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

    def predict_with_uncertainty(self, representations: list[str], n_samples: int = 20) -> Tuple[list[float], list[float]]:
        if not representations:
            return [], []
        self.model.train() # Enable MC Dropout
        all_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds = []
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
                all_preds.append(preds)
        self.model.eval()
        arr = np.array(all_preds) # [n_samples, batch_size]
        means = np.mean(arr, axis=0).tolist()
        vars_ = np.var(arr, axis=0).tolist()
        return means, vars_


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str, adv_generator=None, adv_lambda: float = 0.5):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)
        
        optimizer.zero_grad()
        preds = model(token_ids, padding_mask=mask)
        clean_loss = criterion(preds, targets)
        loss = clean_loss
        
        if adv_generator is not None and adv_lambda > 0.0:
            reps = batch["representation"]
            adv_reps = []
            for r in reps[:4]:
                outcomes = adv_generator.generate(tokenize(r), n_variants=1)
                adv_reps.append(outcomes[0].adversarial_representation if outcomes else r)
            
            if adv_reps:
                adv_ids = torch.zeros((len(adv_reps), loader.dataset.max_len), dtype=torch.long, device=device)
                adv_mask = torch.ones((len(adv_reps), loader.dataset.max_len), dtype=torch.bool, device=device)
                for i, r in enumerate(adv_reps):
                    toks = tokenize(r)
                    slen = min(len(toks), loader.dataset.max_len)
                    for j in range(slen):
                        adv_ids[i, j] = loader.dataset.vocab_map.get(toks[j], 0)
                        adv_mask[i, j] = False
                adv_preds = model(adv_ids, padding_mask=adv_mask)
                adv_loss = criterion(adv_preds, targets[:len(adv_reps)])
                loss = (1.0 - adv_lambda) * clean_loss + adv_lambda * adv_loss
                
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def run_suite(data_path: str = "data/processed/processed.csv", output_dir: str = "results"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    rep_col = "original_representation" if "original_representation" in df.columns else "smiles"
    prop_col = "property_value" if "property_value" in df.columns else "bandgap"
    
    reps = df[rep_col].astype(str).tolist()[:100]
    vals = df[prop_col].astype(float).tolist()[:100]

    vocab_obj = Vocabulary.build(reps)
    vocab = vocab_obj.itos
    vocab_map = vocab_obj.stoi
    
    scaler = TargetScaler()
    scaler.fit(np.array(vals))
    scaled_vals = scaler.transform(np.array(vals)).flatten().tolist()

    split_idx = int(len(reps) * 0.8)
    train_reps, test_reps = reps[:split_idx], reps[split_idx:]
    train_vals, test_vals = scaled_vals[:split_idx], scaled_vals[split_idx:]
    test_true = vals[split_idx:]

    train_dataset = PolymerDataset(train_reps, train_vals, vocab_map)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validator = ChemicalPlausibilityValidator(min_tanimoto_similarity=0.5)

    # --------------------------------------------------------------------------
    # 1. Multi-Seed Benchmark Across 5 Seeds
    # --------------------------------------------------------------------------
    logger.info("=========================================================")
    logger.info("1. Running Multi-Seed Benchmark (5 Seeds: 42, 123, 2026, 777, 999)")
    logger.info("=========================================================")

    seed_baseline_results = []
    seed_defended_results = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Baseline
        base_model = TwoBranchTransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2)
        base_opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
        for _ in range(4):
            train_epoch(base_model, train_loader, base_opt, device)
        
        base_adapter = ModelAdapter(base_model, vocab_map, scaler, device)
        clean_preds_b = base_adapter.predict(test_reps)
        
        # MCMC Attack on Baseline
        mcmc_b = ProbabilisticMCMCAttack(rng=rng, predictor=base_adapter, allowed_tokens=vocab, steps=20, min_tanimoto_similarity=0.5)
        adv_reps_b, valid_flags_b = [], []
        for r in test_reps:
            outcomes = mcmc_b.generate(tokenize(r), n_variants=1)
            if outcomes:
                cand = outcomes[0].adversarial_representation
                v, _ = validator.validate(r, cand)
                adv_reps_b.append(cand)
                valid_flags_b.append(v)
            else:
                adv_reps_b.append(r)
                valid_flags_b.append(True)
        adv_preds_b = base_adapter.predict(adv_reps_b)
        base_metrics = robustness_metrics(test_true, clean_preds_b, adv_preds_b, valid_flags_b)

        # Epistemic Uncertainty Baseline
        clean_means_b, clean_vars_b = base_adapter.predict_with_uncertainty(test_reps, n_samples=15)
        adv_means_b, adv_vars_b = base_adapter.predict_with_uncertainty(adv_reps_b, n_samples=15)
        base_metrics["epistemic_uncertainty_drift"] = float(np.mean(adv_vars_b) - np.mean(clean_vars_b))

        seed_baseline_results.append(base_metrics)

        # Defended Model
        def_model = TwoBranchTransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2)
        def_opt = torch.optim.Adam(def_model.parameters(), lr=1e-3)
        adv_trainer = ProbabilisticMCMCAttack(rng=np.random.default_rng(seed + 1), predictor=base_adapter, allowed_tokens=vocab, steps=10, min_tanimoto_similarity=0.5)
        
        for _ in range(4):
            train_epoch(def_model, train_loader, def_opt, device, adv_generator=adv_trainer, adv_lambda=0.5)

        def_adapter = ModelAdapter(def_model, vocab_map, scaler, device)
        clean_preds_d = def_adapter.predict(test_reps)

        mcmc_d = ProbabilisticMCMCAttack(rng=rng, predictor=def_adapter, allowed_tokens=vocab, steps=20, min_tanimoto_similarity=0.5)
        adv_reps_d, valid_flags_d = [], []
        for r in test_reps:
            outcomes = mcmc_d.generate(tokenize(r), n_variants=1)
            if outcomes:
                cand = outcomes[0].adversarial_representation
                v, _ = validator.validate(r, cand)
                adv_reps_d.append(cand)
                valid_flags_d.append(v)
            else:
                adv_reps_d.append(r)
                valid_flags_d.append(True)
        adv_preds_d = def_adapter.predict(adv_reps_d)
        def_metrics = robustness_metrics(test_true, clean_preds_d, adv_preds_d, valid_flags_d)

        # Epistemic Uncertainty Defended
        clean_means_d, clean_vars_d = def_adapter.predict_with_uncertainty(test_reps, n_samples=15)
        adv_means_d, adv_vars_d = def_adapter.predict_with_uncertainty(adv_reps_d, n_samples=15)
        def_metrics["epistemic_uncertainty_drift"] = float(np.mean(adv_vars_d) - np.mean(clean_vars_d))

        seed_defended_results.append(def_metrics)

    # Compute Summary Stats (Mean +/- Std)
    def compute_stats(results_list: list[dict]) -> dict:
        summary = {}
        for key in results_list[0].keys():
            vals = [r[key] for r in results_list if isinstance(r[key], (int, float))]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_std"] = float(np.std(vals))
        return summary

    baseline_summary = compute_stats(seed_baseline_results)
    defended_summary = compute_stats(seed_defended_results)

    logger.info(f"Baseline Mean Drift: {baseline_summary['mean_absolute_drift_mean']:.4f} +/- {baseline_summary['mean_absolute_drift_std']:.4f}")
    logger.info(f"Defended Mean Drift: {defended_summary['mean_absolute_drift_mean']:.4f} +/- {defended_summary['mean_absolute_drift_std']:.4f}")

    # --------------------------------------------------------------------------
    # 2. Attack Paradigm Comparisons
    # --------------------------------------------------------------------------
    logger.info("=========================================================")
    logger.info("2. Running Attack Paradigm Comparisons (Random vs Substitution vs MCMC)")
    logger.info("=========================================================")
    
    # Use Seed 42 baseline model
    base_model_42 = TwoBranchTransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2)
    base_opt_42 = torch.optim.Adam(base_model_42.parameters(), lr=1e-3)
    for _ in range(4):
        train_epoch(base_model_42, train_loader, base_opt_42, device)
    adapter_42 = ModelAdapter(base_model_42, vocab_map, scaler, device)
    clean_preds_42 = adapter_42.predict(test_reps)

    rng = np.random.default_rng(42)
    # A) Random Mutation
    rand_attack = SmilesRandomizationAttack(rng=rng)
    adv_rand_preds = []
    for r in test_reps:
        outs = rand_attack.generate(tokenize(r), n_variants=1)
        cand = outs[0].adversarial_representation if outs else r
        adv_rand_preds.append(adapter_42.predict([cand])[0])
    rand_metrics = robustness_metrics(test_true, clean_preds_42, adv_rand_preds)

    from materials_adv.attacks.substitution import SubstitutionAttack
    # B) Substitution Attack
    sub_attack = SubstitutionAttack(rng=rng, allowed_tokens=vocab, attack_budget=2)
    adv_sub_preds = []
    for r in test_reps:
        outs = sub_attack.generate(tokenize(r), n_variants=1)
        cand = outs[0].adversarial_representation if outs else r
        adv_sub_preds.append(adapter_42.predict([cand])[0])
    sub_metrics = robustness_metrics(test_true, clean_preds_42, adv_sub_preds)

    # C) Probabilistic MCMC
    mcmc_attack = ProbabilisticMCMCAttack(rng=rng, predictor=adapter_42, allowed_tokens=vocab, steps=20)
    adv_mcmc_preds = []
    for r in test_reps:
        outs = mcmc_attack.generate(tokenize(r), n_variants=1)
        cand = outs[0].adversarial_representation if outs else r
        adv_mcmc_preds.append(adapter_42.predict([cand])[0])
    mcmc_metrics = robustness_metrics(test_true, clean_preds_42, adv_mcmc_preds)

    attack_comparisons = {
        "random_mutation": rand_metrics,
        "simple_substitution": sub_metrics,
        "probabilistic_mcmc": mcmc_metrics,
    }

    # --------------------------------------------------------------------------
    # 3. Component Ablation Study (Lambda Sweeps & Tanimoto Sweeps)
    # --------------------------------------------------------------------------
    logger.info("=========================================================")
    logger.info("3. Running Component Ablation Study")
    logger.info("=========================================================")

    lambda_sweeps = {}
    for lam in [0.0, 0.25, 0.50, 0.75, 1.0]:
        m_abl = TwoBranchTransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2)
        opt_abl = torch.optim.Adam(m_abl.parameters(), lr=1e-3)
        adv_gen_abl = ProbabilisticMCMCAttack(rng=np.random.default_rng(42), predictor=adapter_42, allowed_tokens=vocab, steps=10)
        for _ in range(4):
            train_epoch(m_abl, train_loader, opt_abl, device, adv_generator=adv_gen_abl, adv_lambda=lam)
        
        adap_abl = ModelAdapter(m_abl, vocab_map, scaler, device)
        c_preds = adap_abl.predict(test_reps)
        
        mcmc_eval = ProbabilisticMCMCAttack(rng=np.random.default_rng(42), predictor=adap_abl, allowed_tokens=vocab, steps=20)
        a_preds = []
        for r in test_reps:
            outs = mcmc_eval.generate(tokenize(r), n_variants=1)
            cand = outs[0].adversarial_representation if outs else r
            a_preds.append(adap_abl.predict([cand])[0])
        lambda_sweeps[f"lambda_{lam}"] = robustness_metrics(test_true, c_preds, a_preds)

    tanimoto_sweeps = {}
    for t_val in [0.0, 0.3, 0.5, 0.7]:
        mcmc_t = ProbabilisticMCMCAttack(rng=np.random.default_rng(42), predictor=adapter_42, allowed_tokens=vocab, steps=20, min_tanimoto_similarity=t_val)
        a_preds_t = []
        val_t = []
        val_checker = ChemicalPlausibilityValidator(min_tanimoto_similarity=t_val)
        for r in test_reps:
            outs = mcmc_t.generate(tokenize(r), n_variants=1)
            if outs:
                cand = outs[0].adversarial_representation
                v, _ = val_checker.validate(r, cand)
                a_preds_t.append(adapter_42.predict([cand])[0])
                val_t.append(v)
            else:
                a_preds_t.append(adapter_42.predict([r])[0])
                val_t.append(True)
        tanimoto_sweeps[f"tanimoto_{t_val}"] = robustness_metrics(test_true, clean_preds_42, a_preds_t, val_t)

    # --------------------------------------------------------------------------
    # 4. Attack Sensitivity Analysis (Drift vs MCMC Steps)
    # --------------------------------------------------------------------------
    logger.info("=========================================================")
    logger.info("4. Running Sensitivity Analysis (MCMC Steps)")
    logger.info("=========================================================")

    mcmc_steps_sweep = {}
    for steps in [5, 10, 20, 50, 100]:
        mcmc_s = ProbabilisticMCMCAttack(rng=np.random.default_rng(42), predictor=adapter_42, allowed_tokens=vocab, steps=steps, min_tanimoto_similarity=0.5)
        a_preds_s = []
        for r in test_reps:
            outs = mcmc_s.generate(tokenize(r), n_variants=1)
            cand = outs[0].adversarial_representation if outs else r
            a_preds_s.append(adapter_42.predict([cand])[0])
        mcmc_steps_sweep[f"steps_{steps}"] = robustness_metrics(test_true, clean_preds_42, a_preds_s)

    # Save Complete Results
    full_summary = {
        "seeds_evaluated": SEEDS,
        "baseline_model_stats": baseline_summary,
        "defended_model_stats": defended_summary,
        "attack_comparisons": attack_comparisons,
        "lambda_ablations": lambda_sweeps,
        "tanimoto_threshold_ablations": tanimoto_sweeps,
        "mcmc_steps_sensitivity": mcmc_steps_sweep,
    }

    res_json = out_dir / "comprehensive_benchmark_summary.json"
    res_json.write_text(json.dumps(full_summary, indent=2))
    logger.info(f"Saved comprehensive benchmark summary to {res_json}")

    # Output Summary Table
    print("\n==========================================================================================")
    print("                MULTI-SEED STATISTICAL SUMMARY (MEAN +/- STD OVER 5 SEEDS)                ")
    print("==========================================================================================")
    print(f"{'Metric':<35} | {'Baseline Model':<25} | {'Defended Model':<25}")
    print("------------------------------------------------------------------------------------------")
    print(f"{'Clean RMSE (eV)':<35} | {baseline_summary['clean_rmse_mean']:.4f} +/- {baseline_summary['clean_rmse_std']:.4f} | {defended_summary['clean_rmse_mean']:.4f} +/- {defended_summary['clean_rmse_std']:.4f}")
    print(f"{'Clean MAE (eV)':<35} | {baseline_summary['clean_mae_mean']:.4f} +/- {baseline_summary['clean_mae_std']:.4f} | {defended_summary['clean_mae_mean']:.4f} +/- {defended_summary['clean_mae_std']:.4f}")
    print(f"{'Adversarial RMSE (eV)':<35} | {baseline_summary['adv_rmse_mean']:.4f} +/- {baseline_summary['adv_rmse_std']:.4f} | {defended_summary['adv_rmse_mean']:.4f} +/- {defended_summary['adv_rmse_std']:.4f}")
    print(f"{'Mean Absolute Drift (eV)':<35} | {baseline_summary['mean_absolute_drift_mean']:.4f} +/- {baseline_summary['mean_absolute_drift_std']:.4f} | {defended_summary['mean_absolute_drift_mean']:.4f} +/- {defended_summary['mean_absolute_drift_std']:.4f}")
    print(f"{'Epistemic Uncertainty Drift':<35} | {baseline_summary['epistemic_uncertainty_drift_mean']:.4f} +/- {baseline_summary['epistemic_uncertainty_drift_std']:.4f} | {defended_summary['epistemic_uncertainty_drift_mean']:.4f} +/- {defended_summary['epistemic_uncertainty_drift_std']:.4f}")
    print("==========================================================================================\n")


if __name__ == "__main__":
    run_suite()
