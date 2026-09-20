# SMC Early Stop Audit (Phase 2)

## Finding
SMC seeds 20260921 and 20260922 in Phase 2 exited after Round 1 instead of completing all 3 rounds (0, 1, 2). Despite this, they were marked as `RUN_COMPLETE` in `run_summary.json`.

## Root Cause Analysis
1. The `run_phase2_pilot.py` script has a fixed loop `for round_id in range(start_round, N_ROUNDS):` with `N_ROUNDS = 3`. There is no `break` or early return logic inside this loop.
2. If `run_summary.json` was updated with `"status": "RUN_COMPLETE"`, it means either the script reached the very end naturally (which implies `round_id = 2` executed and completed) OR an external script forcibly patched the JSON status later.
3. Review of the Phase 2 workflow reveals that a patch script (`patch_main.py`) was used to forcefully mark all non-failed runs as `RUN_COMPLETE` to enable aggregation. The SMC runs likely terminated prematurely due to an unhandled exception (e.g. empty candidate generation, matrix inversion error in validation, or OOM) or slurm timeout. Because the patch script blinded the actual completion count, it masked the failure.

## Phase 2B Resolution
To prevent this in Phase 2B:
1. `run_carry_reset.py` will implement strict completion semantics: `RUN_COMPLETE_FULL`, `RUN_PARTIAL`, and `RUN_FAILED`.
2. The number of expected vs completed rounds will be explicitly logged in `run_summary.json`.
3. If an attacker proposes 0 candidates, it will be caught and explicitly handled rather than silently crashing the script or being papered over.
