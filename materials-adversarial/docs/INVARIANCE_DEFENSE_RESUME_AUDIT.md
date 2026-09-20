# Phase 6 Execution State Recovery

## Completed Validation Runs
The following grid combinations completed selection training and have states cached in `results/invariance_defense/lambda_selection.csv`:

- `repeat_shift_average`, seed 20260815, lambdas: 0.01, 0.1, 0.5, 1.0 (COMPLETE)
- `broader_average`, seed 20260815, lambdas: 0.01, 0.1, 0.5, 1.0 (COMPLETE)
- `broader_worst`, seed 20260815, lambdas: 0.01, 0.1, 0.5, 1.0 (COMPLETE)

The best lambda for each arm under the clean validation MAE gate (<=+0.02 eV from baseline) was chosen and its model saved in `results/invariance_defense/models/{arm}/{seed}/model.pt`.

## Interruption Status
The script successfully finished the `select` mode and generated `selection_manifest.json` before being interrupted. No final tests (seeds > 0) have been executed yet.

## Test Seal Status
TEST_SEAL_INTACT
All selection was performed using strictly validation splits. No test files or references to the final Phase 5B/5C sealed cohort appear in the execution path of the `select` mode.
