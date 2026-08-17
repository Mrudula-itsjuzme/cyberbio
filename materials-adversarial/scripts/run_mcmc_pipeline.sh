#!/bin/bash
set -e

echo "=== 1. Training Baseline with Length Gate ==="
PYTHONPATH=src .venv/bin/python -c '
from materials_adv.training.train import train
train(out_dir="results/models/transformer_regressor", write_back_config=False, seed=42)
'

echo "=== 2. Training Exploratory Residualized Model ==="
PYTHONPATH=src .venv/bin/python -c '
from materials_adv.training.train import train
train(out_dir="results/models/transformer_regressor_residualized", write_back_config=False, seed=42, exploratory_residualize=True)
'

echo "=== 3. Running All Attacks (including SMILES Randomization & MCMC) ==="
PYTHONPATH=src .venv/bin/python src/materials_adv/attacks/run_attacks.py --model-dir results/models/transformer_regressor --out-file phase3_baseline_attacks.jsonl
PYTHONPATH=src .venv/bin/python src/materials_adv/attacks/run_attacks.py --model-dir results/models/transformer_regressor_residualized --out-file phase3_residualized_attacks.jsonl

echo "Pipeline finished!"
