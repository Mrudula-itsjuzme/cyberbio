# SMOKE TEST STATUS

## Command Run
`PYTHONPATH=. .venv/bin/python scripts/smoke_test_attack.py`

## Execution Summary
- **Status:** PASS
- **Model / Checkpoint:** `materials-adversarial/results/phase1/model_final.pt` (Baseline TransPolymer)
- **Target Units:** Corrected to `eV` (formerly incorrectly logged as `K`).
- **Sample Count:** 2 test molecules evaluated
- **Attack Used:** `ProbabilisticMCMCAttack` with a budget of 10 evaluation variants each.
- **Validity Constraints:** RDKit parsed, attachment points checked, ring/branch protection enabled.

## Sample Run Result
The model successfully evaluated candidates, discarding variants with invalid representation or heuristically blocked tokens (e.g., token insertion that rdkit could not parse). Drift values (e.g., 0.05 eV, 0.63 eV) were successfully calculated using the MCMC evaluation pipeline. The end-to-end framework remains sound.
