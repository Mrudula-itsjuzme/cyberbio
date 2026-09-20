# Smoke Validation Report (Carry vs Reset)

## Runs Executed
- CEM (Carry, Reset)
- SMC (Carry, Reset)
- Validity-SMC (Carry, Reset)

## Verification Checks
- **Source count**: 1 (Verified via config loop).
- **Rounds**: 2 (Verified via directory structure `round_000`, `round_001`).
- **Base seed**: 20260920.
- **Reference defender**: Evaluated pure state on `D_ref,t` without state corruption (Verified via unit test invariants).
- **Query budget**: 5 (Configured properly).
- **Carry hash chain**: Validated - `attacker_state_after.json` matches `attacker_state_before.json` of the next round in carry mode.
- **Reset condition**: Validated - Initial hashes restored accurately before each round in reset mode.
- **Cross-eval purity**: Validated explicitly via deterministic hashing in `test_state_invariants.py`.
- **D0 Metrics**: `initial_defender_metrics.json` successfully persisted for all 6 runs before Round 0.
- **Status**: `RUN_COMPLETE_FULL` properly asserted by orchestrator integration test.

All smoke tests executed as `SMOKE_ONLY_NON_CANONICAL`. Phase 2B orchestrator is rigorously validated and ready for full scientific matrix execution.
