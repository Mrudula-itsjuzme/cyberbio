#!/bin/bash
set -e
export PYTHONPATH="/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/src"
PYTHON=/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python

SEEDS="20260920 20260921 20260922"
METHODS="cem smc validity_smc"

echo "=== BUILDING REFERENCE TRAJECTORIES ==="
for seed in $SEEDS; do
    echo "Reference seed $seed..."
    $PYTHON scripts/build_reference_trajectory.py --seed $seed --n-sources 10 --rounds 3 --query-budget 25 --smoke false
done

echo "=== RUNNING PAIRED JOBS ==="
for seed in $SEEDS; do
    for method in $METHODS; do
        echo "Running $method seed $seed..."
        $PYTHON scripts/run_carry_reset_pair.py --method $method --seed $seed --rounds 3 --query-budget 25 --n-sources 10 --smoke false
    done
done

echo "=== ALL JOBS COMPLETE ==="
