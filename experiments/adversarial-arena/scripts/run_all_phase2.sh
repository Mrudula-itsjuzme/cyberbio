#!/bin/bash
set -e

SEEDS=(20260920 20260921 20260922)
ATTACKERS=("cem" "smc" "validity_smc")

for attacker in "${ATTACKERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "=================================================="
        echo "Running Phase 2 Pilot: ${attacker} | seed=${seed}"
        echo "=================================================="
        /home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python \
            scripts/run_phase2_pilot.py \
            --attacker "$attacker" \
            --seed "$seed" \
            --results-dir results/arena
    done
done
echo "All Phase 2 pilots completed successfully."
