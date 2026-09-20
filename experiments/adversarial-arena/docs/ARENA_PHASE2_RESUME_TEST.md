# Phase 2 Stop/Resume Test

To verify that the Arena orchestration correctly handles interruptions during probabilistic attacker phases, an explicit Stop/Resume smoke test was conducted.

## Test Procedure
1. A temporary Arena configuration was created using `seed=9999`, a reduced query budget (2 target queries per source), and `N_SOURCES=2`.
2. The `run_phase2_pilot.py` script was invoked for the `cem` attacker and allowed to complete Round 0.
3. The process was forcibly interrupted (`timeout 40s` signal) during Round 1.
4. The same script was invoked again with identical arguments.

## Verification Evidence
The logs and artifacts confirmed the following:

- **Round 0 completed before interruption**: Verified by the presence of `round_000/COMPLETE.json` and a fully populated `round_000` directory before the interruption.
- **Execution stopped**: The `timeout` successfully killed the process mid-way through Round 1.
- **Second invocation detected existing state**: The script logic correctly scanned the output directory.
- **Log contains "Resuming from round 1"**: The stdout from the resumed run explicitly printed `Resuming from round 1`.
- **Round 0 timestamps/artifacts unchanged**: The timestamps for `round_000` artifacts were verified as unmodified during the second execution.
- **Attacker state restored**: CEM loaded `probs` and `entropy` from `round_000/attacker_state_after.json` and used them to initialize Round 1.
- **Defender checkpoint restored**: The `TransPolymerDefender` loaded the weights from `round_000/defender_checkpoint.json`.
- **Hard-example buffer restored**: Verified via `run_summary.json` state propagation.
- **Round 1 completed**: The resumed execution successfully generated `round_001` and printed `Phase 2 pilot complete: cem seed=9999`.
- **No duplicated rows/artifacts**: The final summary and CSVs contained exactly 2 rounds worth of data with no duplicated Round 0 lines.

## Conclusion
The Stop/Resume capability operates correctly for Phase 2 probabilistic attackers. State carry-over and checkpoint restoration have been strictly verified.
