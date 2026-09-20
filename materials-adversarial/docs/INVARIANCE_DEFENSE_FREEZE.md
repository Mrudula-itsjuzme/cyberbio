# Phase 6 Freeze: Invariance Defense Causal Claims

## 1. Clean vs. Robustness Cost
- **Claim**: `ROBUSTNESS_IMPROVED_NO_CLEAN_COST`
- **Evidence**: `broader_worst` achieved massive robustness improvements (1.014 eV reduction, 60.2%) while simultaneously improving Clean MAE (0.4619 → 0.4539 eV, −0.0080 eV) against a strict apples-to-apples baseline trained on the exact same token grammar. *(Note: the value 0.3229 that appeared in an earlier draft of this document was a data entry error — it was a per-source worst-case drift statistic, not a Clean MAE. See `docs/FINAL_METRIC_PROVENANCE.md` for full audit.)*

## 2. Invariance Training vs. Input Pipeline
- **Claim**: 100% of the robustness gain comes from the consistency loss.
- **Evidence**: Correcting the tokenizer bug actually slightly *worsened* the historical baseline's worst-case broader drift (1.563 eV -> 1.685 eV) because attachment points were correctly differentiated instead of masked to `<unk>`. The defense then reduced this 1.685 eV to 0.671 eV.

## 3. Threat Model Efficacy (Paired Statistics)
- **Consistency vs. Augmentation**: `broader_average` outperforms `ordinary_augmentation` (Mean reduction: 0.312 eV, p=3.90e-18, Effect size: 1.63).
- **Broader vs. Repeat/Shift**: `broader_average` outperforms `repeat_shift_average` on broader equivalence (Mean reduction: 0.220 eV, p=3.69e-06, Effect size: 0.61).
- **Worst-case vs. Average-case**: `broader_worst` outperforms `broader_average` on broader equivalence (Mean reduction: 0.236 eV, p=4.67e-18, Effect size: 1.59).

## 4. Generalization
- **Claim**: `GENERALIZES_TO_HELD_OUT_TRANSFORMATION_FAMILY`
- **Evidence**: `branch_order_permutation` was strictly held-out from the training and validation banks. The `broader_worst` model reduced drift on this unseen family from 0.95 eV (baseline) to 0.44 eV.

## 5. Mechanism Evidence
- **Claim**: `POSITIONAL_ENCODING_MECHANISM_EVIDENCE`
- **Evidence**: Ablating positional encodings (`broader_average_no_position`) improved robustness on broader equivalence (0.518 eV vs 0.792 eV), providing evidence that positional lock-in is a primary mechanism for representation divergence.

## Final Primary Effect
- **Model**: `broader_worst` (lambda=1.0)
- **Absolute Drift Reduction**: 1.014 eV
- **Relative Drift Reduction**: 60.18%
- **Median Paired Reduction**: 0.945 eV
- **Fraction of Sources Improved**: 100%

