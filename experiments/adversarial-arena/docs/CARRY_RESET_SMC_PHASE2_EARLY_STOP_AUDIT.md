# SMC Early Stop Audit (Phase 2)

## Finding
SMC seeds 20260921 and 20260922 in Phase 2 exited after Round 1 instead of completing all 3 rounds (0, 1, 2). Despite this, they were marked as `RUN_COMPLETE` in `run_summary.json`.

## Root Cause Analysis
**ROOT_CAUSE = UNKNOWN**

Confirmed facts:
- The Phase 2 configured loop expected 3 rounds.
- No intended scientific early-stop path exists in `run_phase2_pilot.py` for SMC.
- Two SMC runs persist only R0/R1.
- The `run_summary.json` summaries later contain `RUN_COMPLETE`.
- No definitive repository evidence identifies why the loop terminated early.

## Phase 2B Resolution
To definitively prevent ambiguity in Phase 2B, `run_carry_reset.py` will implement robust completion semantics:
- `RUN_COMPLETE_FULL`
- `RUN_COMPLETE_EARLY_STOP`
- `RUN_PARTIAL`
- `RUN_FAILED`
