#!/bin/bash
set -e
export PYTHONPATH="/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/src"
PYTHON=/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python

echo "Building real reference trajectory for smoke..."
$PYTHON scripts/build_reference_trajectory.py --seed 20260920 --n-sources 1 --rounds 2 --query-budget 5

for method in cem smc validity_smc; do
    echo "Running paired smoke for $method"
    $PYTHON scripts/run_carry_reset_pair.py --method $method --seed 20260920 --rounds 2 --query-budget 5 --n-sources 1
done

echo "Smoke matrix complete."
