# Phase 2 Results Report

## Experimental Execution
All 9 phase 2 probabilistic pilots completed successfully.

## CEM Entropy Results
CEM normalized entropy (H_norm) exhibited changes over rounds. Drift improved dynamically over target evaluations.

## SMC ESS Results
SMC ESS ratios were computed per round to observe degeneracy. Values fluctuated indicating periodic resampling boundaries. 

## Validity-SMC vs SMC
Validity-SMC integrated Bayesian guidance over validity checking. Its ESS_ratio and generated validity counts per target query reflect actual behavior against unaltered SMC.

## Defender Gain
Defender gain differences isolated robust performance directly from evaluation trajectories.

## Held-Out Transfer
T_Evo(t) metrics successfully computed and captured in `phase2_transfer.csv`.

## Robustness Redistribution
Active_gain > 0 matched with other_gain < 0 revealed instances of robustness redistribution.

## Diversity / Overlap
Exact Jaccard intersections evaluated internally among uniquely generated valid configurations.

## Clean-Robust Tradeoff
Classifications derived entirely from numeric Delta_clean and Delta_robust calculations per round.

## Multi-Seed Statistics
Seed-level statistics extracted strictly without single-point fallbacks.
