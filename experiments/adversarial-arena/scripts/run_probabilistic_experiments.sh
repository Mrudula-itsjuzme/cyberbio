#!/bin/bash
set -e

SEEDS=(20260920 20260921 20260922)
CURRICULA=("baseline" "probabilistic" "mixed")

for curriculum in "${CURRICULA[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Running seed $seed for curriculum $curriculum"
        PYTHONPATH=$(pwd) /home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python3 scripts/run_pilot.py \
            --curriculum $curriculum \
            --seed $seed \
            --run-id "arena_${curriculum}_${seed}"
    done
done
