# Cross-Attack Matrix

**Artifacts**:
- `results/comparison/experiments/defense_transfer/defense_inventory.csv`
- `results/comparison/experiments/frozen_candidate_banks/*.jsonl`
- `results/materials/experiments/phase4_controlled_robustness/phase_e_f_results/robustness_matrix.csv`

**Findings**:
We extracted frozen banks for Random, MCMC, and Evolutionary attacks (500 candidates each).
We successfully cross-evaluated them against multiple trained checkpoints.
Results show baseline performance degradation under attack, with randomized smoothing improving robustness on structural changes.
