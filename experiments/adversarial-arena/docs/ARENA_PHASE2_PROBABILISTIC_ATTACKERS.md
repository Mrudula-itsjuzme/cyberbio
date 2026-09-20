# Phase 2: Probabilistic Attackers Integration Report

## 1. Query Accounting Protocol
- **Clean Reference Convention**: The clean source evaluation was factored out so it does not decrement the target model budget.
- **Strict Envelope**: `_ArenaPredictor` enforces exactly 25 internal search target queries per source. This brings the evaluation of all internal search algorithms exactly into parity with the fixed budget available to RandomSMILES and MCMC.

## 2. True Adaptation and State Persistence
- **CEM & SMC**: The underlying implementation for CEM and SMC inside `probabilistic-search-lab` does not natively support initializing or carrying state across `run()` boundaries. Therefore, their state drops to `reset` at every single source and round. We label their metric as `NO_PERSISTENT_ADAPTATION_STATE`.
- **Validity-SMC**: The empirical validity surrogate explicitly carries over and updates its Beta posteriors (e.g., Substitution: Beta(alpha, beta)) across sequences and rounds, so `state_mode="carry"` is fully implemented and tested.

## 3. Fixed Benchmark Fairness
- Under the exact same budget (25 queries) and strict RDKit + canonical constraint validation:
  - RandomSMILES & MCMC achieved 0 valid candidates.
  - CEM & SMC generated large numbers of valid, high-drift proposals (e.g., CEM: 240 valid across 10 sources). 
  - This demonstrates that without target-guided search or empirical validity tracking, strict constraints limit naive perturbations to near-zero productivity in small-budget environments.

## 4. Current Status
The 3 curricula (baseline, probabilistic, mixed) × 3 seeds (20260920, 20260921, 20260922) are currently running in the background with the corrected protocol. 
Once completed, aggregation scripts will generate the final cross-training and failure-overlap metrics.
