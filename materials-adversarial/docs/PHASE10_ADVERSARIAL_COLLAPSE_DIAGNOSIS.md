> [!NOTE]
> **Status**: CANONICAL

# Phase 10: Adversarial Collapse Diagnosis

## Goal
Phase 9 demonstrated that adversarial training on worst-case equivalent polymer representations caused catastrophic collapse of the primary regression task (clean R² dropped from 0.82 to 0.14). Phase 10 ablates the training dynamics to isolate the root cause of this failure.

## Ablation Variants
- **A_FailedOriginal**: High adversarial weight (λ=0.1), worst-case adversaries.
- **B_CleanRandom**: High adversarial weight (λ=0.1), random (non-worst-case) canonical equivalents.
- **C_CleanLowAdv**: Low adversarial weight (λ=0.01), worst-case adversaries.
- **D_Curriculum**: Ramping adversarial weight (λ=0.0 to 0.1), worst-case adversaries.
- **E_FrozenEncoder**: Frozen Transformer encoder (λ=0.1), training only the regression head on clean/adversarial pairs.

## Results

### 1. Primary Task Preservation (Regression Performance)
| Variant | Clean MAE | Clean R² | Clean Prediction Shift (vs D0) |
| --- | --- | --- | --- |
| D0 (Baseline) | ~0.450 | ~0.820 | 0.000 |
| **A_FailedOriginal** | 0.686 | 0.612 | 0.572 eV |
| **B_CleanRandom** | 0.685 | 0.613 | 0.570 eV |
| **C_CleanLowAdv** | 0.682 | 0.616 | 0.566 eV |
| **D_Curriculum** | 0.681 | 0.617 | 0.565 eV |
| **E_FrozenEncoder** | **0.469** | **0.801** | **0.170 eV** |

### 2. Adversarial Robustness (Fresh Attack Drift)
*Note: D0 fresh drift baseline is 0.606.*
| Variant | Fresh Drift | Seen Drift |
| --- | --- | --- |
| **A_FailedOriginal** | 0.553 | 0.983 |
| **B_CleanRandom** | 0.554 | 0.986 |
| **C_CleanLowAdv** | 0.559 | 0.992 |
| **D_Curriculum** | 0.560 | 0.993 |
| **E_FrozenEncoder** | 0.603 | 1.064 |

### 3. Embedding and Parameter Drift
- **A through D (Unfrozen Variants):** Exhibited massive parameter drift in the encoder (~0.32) and increased embedding pairwise similarity (from 0.408 in D0 to ~0.461).
- **E (Frozen Encoder):** 0.0 encoder parameter drift (by definition), zero embedding collapse, but failed to improve robustness.

## Hypothesis Verdicts

* **H1 (Excessive Weight): NOT SUPPORTED.** Reducing λ to 0.01 (Variant C) failed to restore clean R² > 0.70.
* **H2 (Catastrophic Forgetting): SUPPORTED.** Even with 75% clean data and low weight, clean prediction shift remained massive (>0.56 eV) when the encoder was allowed to update.
* **H3 (Worst-Case Bias): NOT SUPPORTED.** Using random representations instead of worst-case (Variant B) did not prevent collapse (R² = 0.613).
* **H4 (Frozen Encoder Preserves Clean Performance): SUPPORTED.** Freezing the encoder (Variant E) preserved clean performance (R² = 0.801).
* **H5 (Curriculum Avoids Collapse): NOT SUPPORTED.** Ramping the weight (Variant D) did not prevent collapse.
* **H6 (Random is Stable): NOT SUPPORTED.** Random adversaries still triggered collapse.

## Verdict and Recommendation

**The tested post-hoc fine-tuning strategies did not achieve fresh representation invariance without unacceptable degradation of Bandgap regression.**

* When the encoder is allowed to update (Variants A-D), the Transformer aggressively rewires its embeddings to minimize the adversarial penalty, completely destroying the finely-tuned chemical manifold required for accurate bandgap regression.
* When the encoder is frozen (Variant E), the regression head cannot bridge the gap between divergent canonical equivalents because the fundamental representations being emitted by the encoder are too far apart. The regression head alone cannot "fix" representation variance.

The fundamental issue is that the current standard SMILES Transformer architecture treats canonically equivalent strings as completely distinct sequences, and forcing it to map them to the exact same scalar output creates devastating gradients that tear apart the latent space.

**Next Step Recommendation:**
As per user instructions ("If all variants fail... recommend Phase 11: REPRESENTATION-INVARIANT ARCHITECTURE COMPARISON"), we must proceed to Phase 11. 
We must abandon forcing a standard Transformer to learn invariance via fine-tuning, and instead implement an architecture that is *structurally invariant* (e.g. Graph Neural Networks, or SMILES data augmentation applied from epoch 0).
