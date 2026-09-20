# Attacker-defender loop

Each round persists `RUNNING_ATTACK → VALIDATING → UPDATING_DEFENDER → EVALUATING
→ ROUND_COMPLETE`. Candidate validation is authoritative and tri-state; unchecked
is never promoted to valid. Zero valid candidates remain a recorded negative
result. A failed update writes `FAILED.json`, leaves the previous checkpoint in
place, and aborts rather than inventing continuation data.

Static adapters currently connect canonical RandomSMILES and model-guided MCMC.
The adaptive component selects between these families; it does not mutate their
implementations. Evolutionary and attribution adapters remain pending.

