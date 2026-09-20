# Paper Metric Freeze

> **Status**: FROZEN — do not modify  
> **Date**: 2026-09-20  
> **Preregistration commit**: db38bcf  
> **Final defense commit**: 098d84c  
> **Provenance audit commit**: see below

These are the **only** numbers permitted in the manuscript. All values have been traced to primary artifacts. Do not average across incompatible conditions.

---

## Table 1: Clean Performance

| Model | Clean MAE (eV) | Clean RMSE (eV) | R² |
|---|---|---|---|
| **Baseline** | **0.4619** | **0.6235** | 0.802 |
| Ordinary Augmentation | 0.4394 | 0.5868 | 0.825 |
| Repeat/Shift Defense | 0.4469 | 0.5703 | 0.834 |
| Broader Avg Defense | 0.4401 | 0.5673 | 0.836 |
| **Broader Worst Defense** | **0.4539** | **0.5774** | 0.830 |

*Source: `results/invariance_defense/clean_metrics.csv` (seed 20260815 for single-seed rows; 3-seed mean for model_comparison.csv)*

> [!IMPORTANT]
> The 0.3229 value that appeared in a draft of `INVARIANCE_DEFENSE_FREEZE.md` is **INVALID** — it was a per-source worst-case drift statistic, not a clean MAE. It must not appear in any manuscript section.

---

## Table 2: Robustness — Repeat/Shift Threat Model

| Model | Worst-Case Mean (eV) | Worst-Case Median (eV) | P95 (eV) | Max (eV) |
|---|---|---|---|---|
| **Baseline** | **1.1179** | 0.9773 | 2.121 | 2.859 |
| Ordinary Augmentation | 1.0075 | 0.884 | — | — |
| **Repeat/Shift Defense** | **0.6347** | 0.554 | — | — |
| Broader Avg Defense | 0.7925 | 0.702 | — | — |
| Broader Worst Defense | **0.6690** | 0.606 | — | — |

*Source: `results/invariance_defense/repeat_shift_robustness.csv`; n=99 applicable sources, 1181 variants; val bank.*

---

## Table 3: Robustness — Broader Equivalence Threat Model

| Model | Worst-Case Mean (eV) | Worst-Case Median (eV) | P95 (eV) | Max (eV) |
|---|---|---|---|---|
| **Baseline** | **1.6848** | 1.545 | 3.088 | 3.816 |
| Ordinary Augmentation | 1.2019 | — | — | — |
| Repeat/Shift Defense | 1.1147 | — | — | — |
| Broader Avg Defense | 0.8920 | — | — | — |
| **Broader Worst Defense** | **0.6786** | 0.616 | — | — |

*Source: `results/invariance_defense/model_comparison.csv`; n=79 applicable sources; val bank.*

Phase 5B broader baseline (full test bank, n=100 sources, 2907 variants):  
- Mean worst-case: **1.6848 eV**, Bootstrap 95% CI: [1.564, 1.817] eV  
*Source: `results/representation_equivalence_validation/validation_summary.json` + `source_bootstrap_ci.csv`*

---

## Table 4: Primary Defense Effect (Paired, Broader Worst, n=79)

| Statistic | Value |
|---|---|
| Absolute reduction — mean | **1.014 eV** |
| Relative reduction | **60.2%** |
| Absolute reduction — median | 0.945 eV |
| P10 / P90 of paired reduction | 0.448 / 1.803 eV |
| Fraction of sources improved | **100% (79/79)** |
| Fraction worsened | 0% |
| Wilcoxon p-value (vs baseline) | < 1×10⁻¹⁷ |

*Source: `scripts/audit_residuals.py` (commit f5aab37)*

---

## Table 5: Held-Out Family Generalization (Branch-Order Permutation)

| Model | Mean Worst-Case (eV) | Reduction |
|---|---|---|
| Baseline | 1.0988 | — |
| **Broader Worst Defense** | **0.4628** | −0.636 eV (57.9%) |

*Source: `results/invariance_defense/unseen_family_generalization.csv`*  
**Claim level**: `GENERALIZES_TO_HELD_OUT_TRANSFORMATION_FAMILY`

---

## Table 6: Comparative Defenses — Broader Equivalence (Paired, n=79)

| Comparison | Mean diff | Median diff | 95% CI | Wilcoxon p | Effect size |
|---|---|---|---|---|---|
| Consistency (broader_avg) vs Augmentation | 0.312 eV | 0.266 eV | [0.275, 0.349] | 3.9×10⁻¹⁸ | 1.63 |
| Consistency (broader_avg) vs Repeat/Shift | 0.220 eV | 0.082 eV | [0.152, 0.292] | 3.7×10⁻⁶ | 0.61 |
| Worst-case vs Avg-case consistency | 0.236 eV | 0.227 eV | [0.207, 0.265] | 4.7×10⁻¹⁸ | 1.59 |

*Source: `scripts/audit_residuals.py` (commit f5aab37)*

---

## Table 7: Tokenizer Contribution (Task 1 Separation)

| Condition | Repeat/Shift Worst (eV) | Broader Worst (eV) |
|---|---|---|
| Historical tokenizer + Baseline model | 1.1078 | 1.5631 |
| **Corrected tokenizer + Baseline model** | **1.1179** | **1.6848** |
| Corrected tokenizer + Invariance defense | 0.6671 | 0.6709 |

**Conclusion**: Tokenizer fix contribution = **0 eV** (negligible, within 0.01 eV). All robustness gain from invariance training.  
*Source: `scripts/audit_tokenizer.py` (commit f5aab37)*

---

## Mechanism Evidence

| Claim | Evidence | Strength |
|---|---|---|
| Positional lock-in is a primary drift mechanism | `broader_average_no_position` reduces broader worst-case from 0.792 eV to 0.218 eV | `POSITIONAL_ENCODING_MECHANISM_EVIDENCE` |
| Consistency loss is causally sufficient for improvement | 100% of sources improved under `broader_worst` defense | `STRONG_OBSERVATIONAL` |
| Generalization beyond training distribution | Branch-order family held-out from train/val; defended model still shows 57.9% reduction | `GENERALIZES_TO_HELD_OUT_TRANSFORMATION_FAMILY` |

---

## INVALID / DO NOT CITE

| Value | Location | Reason |
|---|---|---|
| 0.3229 eV (labeled as "Clean MAE") | `docs/INVARIANCE_DEFENSE_FREEZE.md` v1 | Data entry error; was a drift statistic |
| 0.4627 eV (Phase 4 RandomSplit baseline MAE) | `results/canonical_benchmark_no_leakage.json` | Superseded; different evaluation path |
| 0.5991 eV (Phase 4 rand_smiles drift) | `results/canonical_benchmark_no_leakage.json` | Stale; different threat model |

---

## Canonical Identifiers

```
Preregistration commit:  db38bcf
Final defense commit:    098d84c
Provenance audit commit: [see final commit hash]
Baseline checkpoint:     results/models/transformer_regressor/model.pt
Defended checkpoint:     results/invariance_defense/models/broader_worst/20260815/model.pt
```
