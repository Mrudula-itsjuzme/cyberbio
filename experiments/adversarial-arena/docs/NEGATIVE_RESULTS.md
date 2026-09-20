# Negative Results & Limitations

- **D0 Clean Limitation**: Initial clean metric D0 was evaluated in memory but not persisted. Round 0 clean-performance delta is NOT_AVAILABLE.
- **Negative G_adapt Instances**: G_adapt was mostly positive but experienced capability drops (< 0) in 4 of the 25 rounds (CEM R2, SMC R1/R2, Validity-SMC R1). Furthermore, defender gains were frequently negative (SMC round 0 mean = -0.6373).
- **CEM Concentration**: Severe collapse (H_norm < 0.20) was encountered in 4 out of 9 CEM trajectories, severely bottlenecking dimensional exploration.
- **Validity-SMC Tradeoff**: Validity-SMC generated more valid proposals per query (mean diff +0.2616), but yielded negative differences in drift per query (mean diff -0.0698). The efficiency benefit is constrained entirely by target objectives.
- **Negative Held-Out Transfer**: Evaluated via Evolutionary attacker cross-evaluations. SMC yielded negative transfer in 2 of 3 evaluable iterations, CEM in 2 of 3 iterative bounds.
- **Robustness Redistribution**: 1 distinct empirical redistribution event captured where active robustness improved while out-of-distribution robustness decayed.
- **Clean Degradation**: `ROBUST_WIN_CLEAN_COST` and `ROBUST_LOSS_CLEAN_COST` were actively monitored and accounted for ~50% of the classifiable training trajectory updates.
