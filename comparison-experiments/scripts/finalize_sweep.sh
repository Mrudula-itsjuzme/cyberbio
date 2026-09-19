#!/bin/bash
set -e

ENV_BIN="/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python"

echo "Aggregating Budget Sweep Results..."
$ENV_BIN scripts/aggregate_budget_sweep.py

echo "Computing Statistics..."
$ENV_BIN scripts/compute_sweep_statistics.py

echo "Generating Plots..."
$ENV_BIN scripts/plot_budget_sweep.py

echo "Sweep Finalization Complete."
