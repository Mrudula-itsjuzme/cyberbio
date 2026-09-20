# Phase 2 Results Report

## D0 Clean Metric Limitation
Initial clean metric D0 was evaluated in memory but not persisted. Therefore: Round 0 clean-performance delta is NOT_AVAILABLE. Clean/robust tradeoff begins only once consecutive persisted defender checkpoints exist.

## G_adapt (Internal Adaptation)
20 of 25 measured seed-round events produced positive G_adapt. This is consistent with frequent benefit from within-attacker adaptation under the tested configuration, although adaptation was not uniformly beneficial. 5/25 G_adapt measurements were negative: 
- CEM seed 20260922 round 2
- SMC seed 20260920 round 1
- SMC seed 20260920 round 2
- Validity-SMC seed 20260920 round 1
- Validity-SMC seed 20260920 round 2

(This does NOT establish that carry-state outperforms reset because reset controls were not run).

| method | round | n_seeds | mean | sample_std |
|---|---|---|---|---|
| cem | 0 | 3 | 0.1860 | 0.1160 |
| cem | 1 | 3 | 0.1277 | 0.0694 |
| cem | 2 | 3 | -0.0110 | 0.0902 |
| smc | 0 | 3 | 1.5740 | 0.6299 |
| smc | 1 | 3 | 0.8626 | 0.8416 |
| smc | 2 | 1 | -0.0571 | 0.0000 |
| validity_smc | 0 | 3 | 1.6799 | 0.4427 |
| validity_smc | 1 | 3 | 0.0114 | 0.1175 |
| validity_smc | 2 | 3 | 0.0778 | 0.1376 |

## Defender Gain Difference
Variance is extremely large. Mixed / high-variance defender response observed across methods.

| method | round | n_seeds | mean | sample_std |
|---|---|---|---|---|
| cem | 0 | 3 | -0.3028 | 0.0667 |
| cem | 1 | 3 | 0.0772 | 0.1299 |
| cem | 2 | 3 | -0.0477 | 0.0969 |
| smc | 0 | 3 | -0.6373 | 0.9230 |
| smc | 1 | 3 | 0.1550 | 0.2832 |
| smc | 2 | 1 | -0.4028 | 0.0000 |
| validity_smc | 0 | 3 | -1.7769 | 0.4124 |
| validity_smc | 1 | 3 | 0.2426 | 0.3274 |
| validity_smc | 2 | 3 | -0.2331 | 0.3189 |

## Held-Out Transfer ($T_{Evo}$)
- CEM: weak/mixed transfer
- SMC: mixed transfer
- Validity-SMC: highly heterogeneous transfer

| method | round | n_seeds | mean | sample_std |
|---|---|---|---|---|
| cem | 0 | 3 | -0.0289 | 0.0242 |
| cem | 1 | 3 | 0.0562 | 0.0333 |
| cem | 2 | 3 | -0.0166 | 0.0114 |
| smc | 0 | 1 | -0.0534 | 0.0000 |
| smc | 1 | 1 | 0.0663 | 0.0000 |
| smc | 2 | 1 | -0.0234 | 0.0000 |

## CEM Collapse
Across all 9 CEM events (using 2-family limits):
- severe collapse: 4
- concentration: 3
- broad: 2

## SMC Degeneracy
Across SMC (7 total events) and Validity-SMC (9 total events):
- smc: healthy (7), moderate (0), severe (0)
- validity_smc: healthy (9), moderate (0), severe (0)

## Validity-SMC vs SMC Efficiency
Differences (Validity-SMC minus SMC):
- validity_rate: mean=0.0046, std=0.0095
- valid_per_target_query: mean=0.2616, std=0.2447
- mean_drift: mean=0.2481, std=0.6446
- drift_per_target_query: mean=-0.0698, std=0.1065
- ess_ratio: mean=-0.1559, std=0.1697
- unique_candidates: mean=4.5714, std=4.2762

## Robustness Redistribution
Events observed (`active_gain_difference > 0` and `other_gain_difference < 0`):
- cem: 1
- smc: 0
- validity_smc: 0

## Candidate Overlap (Mean Jaccard)
- CEM vs SMC: 0.0000
- CEM vs Validity-SMC: 0.0058
- SMC vs Validity-SMC: 0.3333

CEM near-zero overlap suggests it explores a substantially different candidate set under this protocol.

## Clean-Robust Tradeoff
- CEM: ROBUST_WIN_CLEAN_WIN (2), ROBUST_WIN_CLEAN_COST (1), ROBUST_LOSS_CLEAN_WIN (1), ROBUST_LOSS_CLEAN_COST (2)
- SMC: ROBUST_WIN_CLEAN_WIN (1), ROBUST_WIN_CLEAN_COST (1), ROBUST_LOSS_CLEAN_WIN (1), ROBUST_LOSS_CLEAN_COST (1)
- Validity-SMC: ROBUST_WIN_CLEAN_WIN (2), ROBUST_WIN_CLEAN_COST (1), ROBUST_LOSS_CLEAN_WIN (1), ROBUST_LOSS_CLEAN_COST (2)
