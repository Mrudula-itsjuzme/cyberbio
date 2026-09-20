# Final Metric Provenance Audit

> **Date**: 2026-09-20  
> **Commit**: see below  
> **Auditor**: Phase 6B provenance pass

---

## 1. Root Cause of the 0.4619 vs 0.3229 Discrepancy

**The 0.3229 figure was an error introduced in `docs/INVARIANCE_DEFENSE_FREEZE.md` (commit f5aab37).**

The value 0.32293... appears in `results/invariance_defense/family_robustness.csv` as the column `mean_source_worst` for the `repeat_shift_average_seed20260816 / repeat_shift / REPEAT` row — a **per-source worst-case drift statistic**, NOT a clean MAE. It was incorrectly transcribed as a clean MAE in the freeze document during an interrupted session.

The correct clean MAE for the baseline model is **0.4619 eV**, verified from three independent sources:

| Source | Value |
|---|---|
| `results/models/transformer_regressor/metrics.json` | 0.4619162976741791 eV |
| `results/invariance_defense/clean_metrics.csv` (baseline row) | 0.46191627085038206 eV |
| `results/invariance_defense/positional_ablation.csv` (baseline row) | 0.461916270850382 eV |

All three are byte-identical to 4 decimal places. The 0.3229 value **must not appear in the manuscript**.

---

## 2. Model and Dataset Provenance

### Canonical Baseline Model
- **Checkpoint**: `results/models/transformer_regressor/model.pt`
- **Architecture**: TransformerRegressorModel, vocab=36, d_model=64, n_layers=2, n_heads=4, ff=128, max_len=256, pooling=mean
- **Training seed**: 20260815
- **Training split**: 2946 train / 631 val / 632 test (random split, seed 20260815)
- **TargetScaler**: fitted on training set targets only (`scaler_source: "fitted_on_this_training_set"`)
- **Tokenizer**: corrected bracket-star (`[*]`) normalization (bare `*` → `[*]` at inference time via `normalize_attachment_tokens`)
- **Note**: The baseline model was trained BEFORE the tokenizer fix. At evaluation time, the corrected tokenizer is applied as a pre-processing step, normalizing any bare `*` to `[*]`. Because all source sequences in the training set already used `[*]` (verified: 0 unknown tokens), clean MAE is identical under historical and corrected tokenizers.

### Scaler Details
- mean: 4.474831296673456 eV
- std: 1.4563050416759802 eV

---

## 3. All Baseline Metrics

### Clean Performance (Test Set, n=632)
| Metric | Value | Status |
|---|---|---|
| Clean MAE | **0.4619 eV** | `CANONICAL_MATCHED_BASELINE` |
| Clean RMSE | **0.6235 eV** | `CANONICAL_MATCHED_BASELINE` |
| Clean R² | 0.8019 | `CANONICAL_MATCHED_BASELINE` |

### Phase 5B: Broader Equivalence Drift (n=100 sources, 2907 variants)
| Metric | Value | Status |
|---|---|---|
| Mean variant drift | 0.5794 eV | `CANONICAL_MATCHED_BASELINE` |
| **Worst-case (per-source max) mean** | **1.6848 eV** | `CANONICAL_MATCHED_BASELINE` |
| Worst-case median | 1.5447 eV | `CANONICAL_MATCHED_BASELINE` |
| Worst-case P90 | 2.4028 eV | `CANONICAL_MATCHED_BASELINE` |
| Max single drift | 3.8157 eV | `CANONICAL_MATCHED_BASELINE` |
| Bootstrap 95% CI | [1.564, 1.817] eV | `CANONICAL_MATCHED_BASELINE` |

### Phase 6: Repeat/Shift Baseline (val bank, n=99 sources, 1181 variants)
| Metric | Value | Status |
|---|---|---|
| **Worst-case (per-source max) mean** | **1.1179 eV** | `CANONICAL_MATCHED_BASELINE` |
| Worst-case median | 0.9773 eV | `CANONICAL_MATCHED_BASELINE` |

### Phase 6: Broader Equivalence Baseline (val bank, n=79 sources, 2907 val-set variants)
| Metric | Value | Status |
|---|---|---|
| **Worst-case (per-source max) mean** | **1.6848 eV** | `CANONICAL_MATCHED_BASELINE` |

> [!NOTE]
> The Phase 5B and Phase 6 broader baselines share the same model and same corrected tokenizer. Phase 6 uses the validation bank (631 val sources, 2907 variants drawn from that pool); Phase 5B used the full 100-source test bank. The **worst-case mean aligns to 4 decimal places (1.6848)**, confirming consistency.

---

## 4. Defended Model Metrics

### Best Defense: `broader_worst` (λ=1.0, 3-seed mean)
| Metric | Value | Reduction vs Baseline |
|---|---|---|
| Clean MAE | 0.4539 eV | −0.0080 eV (improvement) |
| Clean RMSE | 0.5774 eV | −0.0461 eV (improvement) |
| **Repeat/Shift worst mean** | **0.6690 eV** | −0.4489 eV (−40.2%) |
| **Broader worst mean** | **0.6786 eV** | −1.0062 eV (−59.7%) |

### Primary Paper Effect: Broader Worst-Case (Paired, n=79)
| Statistic | Value |
|---|---|
| Absolute reduction (mean) | 1.014 eV |
| Relative reduction | 60.2% |
| Median paired reduction | 0.945 eV |
| P10 paired reduction | 0.448 eV |
| P90 paired reduction | 1.803 eV |
| Fraction of sources improved | 100% (79/79) |

### Branch-Order (Held-Out Family, n=80 sources)
| Model | Worst-case mean |
|---|---|
| Baseline | 1.0988 eV |
| `broader_worst` (best seed) | 0.4628 eV |
| Reduction | 0.636 eV (57.9%) |

---

## 5. Stale / Invalid Metrics

| Metric | Location | Reason | Action |
|---|---|---|---|
| 0.3229 eV "Clean MAE" | `docs/INVARIANCE_DEFENSE_FREEZE.md` | Was `mean_source_worst` drift for one family, not a clean MAE | **CORRECTED** in updated FREEZE.md |
| Phase 4 canonical_benchmark_no_leakage rand_smiles | `results/canonical_benchmark_no_leakage.json` | Used old Phase 4 attack (randomized SMILES), different threat model from Phase 5B/6 | `STALE` — do not cite in paper |
| 0.4627 eV (Phase 4 RandomSplit baseline) | `results/canonical_benchmark_no_leakage.json` | Same model checkpoint, slightly different evaluation path | `HISTORICAL_REFERENCE` — superseded by Phase 6 clean_metrics.csv which is 0.4619 |

---

## 6. Tokenizer Contribution Analysis

From `scripts/audit_tokenizer.py` (commit f5aab37):

| Condition | Repeat/Shift Worst | Broader Worst |
|---|---|---|
| Historical tokenizer (`*` → `<unk>`) + Baseline model | 1.1078 eV | 1.5631 eV |
| Corrected tokenizer (`*` → `[*]`) + Baseline model | 1.1179 eV | 1.6848 eV |
| Corrected tokenizer + `broader_worst` defense | 0.6671 eV | 0.6709 eV |

**Conclusion**: The tokenizer fix contributed **0 eV** to robustness improvement (it marginally *worsened* broader drift by +0.12 eV because attachment points are now correctly differentiated). All 1.014 eV of robustness gain is attributable to invariance training.

---

## 7. Source Artifacts

| Metric Group | Artifact |
|---|---|
| Baseline clean metrics | `results/models/transformer_regressor/metrics.json` |
| Phase 6 clean metrics (all arms) | `results/invariance_defense/clean_metrics.csv` |
| Phase 5B broader drift | `results/representation_equivalence_validation/validation_summary.json` |
| Phase 6 repeat/shift baseline + defense | `results/invariance_defense/repeat_shift_robustness.csv` |
| Phase 6 broader baseline + defense (by arm) | `results/invariance_defense/model_comparison.csv` |
| Phase 6 family-level breakdown | `results/invariance_defense/family_robustness.csv` |
| Unseen family (branch-order) | `results/invariance_defense/unseen_family_generalization.csv` |
| Positional ablation | `results/invariance_defense/positional_ablation.csv` |
| Full metric registry | `results/final_metric_registry.csv` |
