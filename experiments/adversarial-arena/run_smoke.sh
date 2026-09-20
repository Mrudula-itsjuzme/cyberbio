#!/bin/bash
set -e
PYTHON=/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python
for method in cem smc validity_smc; do
  for mode in carry reset; do
    echo "Running smoke for $method $mode"
    $PYTHON scripts/run_carry_reset.py --method $method --mode $mode --seed 20260920 --rounds 2 --query-budget 5 --n-sources 1 --smoke
  done
done
