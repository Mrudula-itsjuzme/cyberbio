# Phase 6 Invariance Defense Selection Freeze

Before evaluating on the final Phase 5B/5C sealed test cohort, we have frozen the defense hyperparameters via a validation-only grid search.

## Selection Criterion
- **Clean Performance Gate**: Configs must have a validation MAE <= 0.480 eV (baseline + 0.02 eV).
- **Robustness Criterion**: Among configs satisfying the gate, the one with the lowest validation mean worst-case drift is selected.
- **Selection Seed**: 20260815 (solely evaluated for selection).

## Selected Defenses

### REPEAT_SHIFT_AVERAGE (repeat_shift)
- **Selected lambda**: 1.0
- **Validation Clean MAE**: 0.447 eV
- **Validation Robustness (drift)**: 0.516 eV
- **Why selected**: Safely passed the clean performance gate and achieved the best repeat/shift robustness.
- **Checkpoint location**: `results/invariance_defense/models/repeat_shift_average/20260815/model.pt`

### BROADER_AVERAGE (broader_average)
- **Selected lambda**: 1.0
- **Validation Clean MAE**: 0.447 eV
- **Validation Robustness (drift)**: 0.457 eV
- **Why selected**: Achieved the best broader-equivalence robustness while remaining under the clean MAE gate limit.
- **Checkpoint location**: `results/invariance_defense/models/broader_average/20260815/model.pt`

### BROADER_WORST (broader_worst)
- **Selected lambda**: 1.0
- **Validation Clean MAE**: 0.467 eV
- **Validation Robustness (drift)**: 0.338 eV
- **Why selected**: Produced the absolute lowest validation drift among all defenses and passed the clean gate limit.
- **Checkpoint location**: `results/invariance_defense/models/broader_worst/20260815/model.pt`

These configurations are now frozen. We will proceed to train the remaining seeds (20260816, 20260817) and evaluate all defenses exactly once on the sealed Phase 5 test sets.
