# Reproducibility Guide

This document captures the exact configurations required to replicate the Phase 1 and Phase 2 experimental results.

## Data Procurement
- **Dataset Source**: `polyVERSE` (Ramprasad Group, Zenodo Record `13352644`).
- **Target File**: `bandgap_chain.csv`
- **Large Unlabelled Pool**: `PI1M.csv` (Note: PI1M is stored locally at `../PI1M.csv`. It is not committed to the repository to prevent git-bloat. Future iterations should dynamically stream this from Figshare).
- **Split Configuration**: Stratified split utilizing `split_dataset.py`.
- **Split Seed**: `20260815`

## Model Configuration (Baseline & Defended)
- **Architecture**: `configs/model.yaml` (Transformer Regressor).
- **Target Feature**: `bandgap_chain` (Target Units: `eV`).
- **Target Scaler**: The standardization scaler `scaler.json` was fitted exclusively on the `2,946` clean Phase 1 training records and explicitly passed to the Phase 2 training loop to enforce comparability.
- **Random Seeds**: Both model training loops used global RNG seed `20260815`.

## Attack Configuration
- **Attack Parameters**: `configs/attack.yaml`
- **Phase 2 Training Budget**: `attack_budget = 1`
- **Phase 2 Target Split**: Generated from the `train` split.
- **Phase 3 Test Split Attack Seed**: `42` (Enforced symmetrical candidate generation for both Baseline and Defended evaluations).

## Evaluation
- **Success Criterion**: Absolute Prediction Drift `> 0.4619 eV` (The established Baseline Test MAE) AND strictly chemically valid (`RDKit == True`).
- **Evaluation Script**: `scripts/eval_phase2_full.py`
- **Output Locations**: 
  - Raw JSONL evaluation metrics for the Test Split are stored in `results/phase2/baseline_adversarial_results.jsonl` and `results/phase2/defended_adversarial_results.jsonl`.
  - Saved model weights are stored in `results/models/transformer_regressor/` and `results/models/transformer_defended/`.
