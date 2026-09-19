# Cross-Attack Matrix

**Artifacts**:
- `comparison-experiments/results/defense_transfer/defense_inventory.csv`
- `comparison-experiments/results/frozen_candidate_banks/*.jsonl`
- `materials-adversarial/results/phase4_controlled_robustness/phase_e_f_results/robustness_matrix.csv`

**Findings**:
We extracted frozen banks for Random, MCMC, and Evolutionary attacks (500 candidates each).
We successfully cross-evaluated them against multiple trained checkpoints.
Results show baseline performance degradation under attack, with randomized smoothing improving robustness on structural changes.
