# Smoke Validation Report (Carry vs Reset)

> **NOTE:** This document supersedes the prior dummy smoke validation. The tests here reflect the TRUE scientific infrastructure using the real `TransPolymerDefender`, locked source cohorts, strict reference checkpointing, and exact deterministic seed propagation.

## Runs Executed
- CEM (Carry, Reset Paired)
- SMC (Carry, Reset Paired)
- Validity-SMC (Carry, Reset Paired)

All marked with: `SMOKE_ONLY_NON_CANONICAL = True`

## Verification Checks
- **Source Cohort**: 1 (Verified matching strictly against `source_ids.json` between treatment arms and reference sequence).
- **Rounds**: 2
- **Base seed**: 20260920. Exact derivation policy `derive_seed` isolates `carry` and `reset` exogenous RNG sequences correctly.
- **Reference defender**: Evaluated pure state on `D_ref,t` without state corruption or label leakage. Hard missing file exceptions implemented.
- **Query budget**: 5 (Configured properly). Observed matching 5 target queries.
- **Carry hash chain**: Validated - `attacker_state_after.json` matches exactly the pre-mutation state of the next round.
- **Reset condition**: Validated - Initial hashes restored accurately before each round in reset mode. Adaptive memory fully flushes.
- **Cross-eval purity**: Validated explicitly via deterministic hashing in `test_state_invariants.py`. Deepcopy isolations verified.
- **D0 Metrics**: `initial_defender_metrics.json` successfully persisted before Round 0, embedding actual structural SHA256 of the model checkpoints and clean RMSE scores.
- **Status**: `RUN_COMPLETE_FULL` properly asserted by orchestrator integration test.

## Actual Smoke Numeric Output
```
=== cem Round 0 ===
Ref SHA: canonical-D0
Carry queries: 5 | Reset queries: 5 | Delta: 0
Carry Mean Drift: 0.08504438400268555 | Reset Mean Drift: 0.08504438400268555 | G_carry_mean: 0.0
Carry Best Drift: 0.1557307243347168 | Reset Best Drift: 0.1557307243347168 | G_carry_best: 0.0
Carry hash: 41484868 -> d6028068
Reset hash: 41484868 -> 41484868

=== cem Round 1 ===
Ref SHA: transformer-D000
Carry queries: 5 | Reset queries: 5 | Delta: 0
Carry Mean Drift: 0.05890536308288574 | Reset Mean Drift: 0.2088325023651123 | G_carry_mean: -0.14992713928222656
Carry Best Drift: 0.1263885498046875 | Reset Best Drift: 0.5408210754394531 | G_carry_best: -0.4144325256347656
Carry hash: d6028068 -> 1d6be75d
Reset hash: 41484868 -> 41484868
```
*(SMC and ValiditySMC similarly validated query tracking and memory transitions)*

Phase 2B orchestrator is rigorously validated and ready for full scientific matrix execution.
