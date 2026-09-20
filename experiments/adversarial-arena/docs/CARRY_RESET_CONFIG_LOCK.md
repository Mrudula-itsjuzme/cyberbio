# Phase 2B: Carry vs Reset Config Lock

## RNG_POLICY
Reset condition:
Restore algorithmic adaptive state to initial prior before each round, but derive per-round RNG deterministically from `base_seed + round_id + source_id`.

Carry condition:
Use the exact same per-round RNG schedule as the reset condition.
This avoids giving carry/reset different randomness merely because their internal RNG consumed different numbers previously.

## REFERENCE DEFENDER DESIGN
TREATMENT-INDEPENDENT REFERENCE DEFENDER TRAJECTORY (`D_ref,t`):
Generated entirely independently of carry/reset assignment.
- source cohort locked
- hard-example rule locked
- defender initialization locked
- training settings locked
- checkpoint sequence locked
Checkpoints persisted to `reference_defender/D_ref_000`, `reference_defender/D_ref_001`, etc.
Both carry/reset attackers must be evaluated against exactly the same D_ref,t checkpoint.
Neither carry nor reset may modify D_ref,t.
