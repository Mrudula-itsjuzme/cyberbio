# Audit Report: Existing Paired Benchmark Artifact

**Target Directory**: `materials-adversarial/results/phase2_paired_benchmark/`  
**Audit Date**: September 12, 2026  
**Auditor**: Automated Forensic Agent  
**Final Classification**: **`VALID BUT DEVELOPMENTAL`**

---

## 1. Provenance & Artifact Integrity

- **Creation Timestamp**: Saturday, September 12, 2026 (11:51:24 IST)
- **Generating Script**: [`scripts/run_phase2_paired_benchmark.py`](../../scripts/run_phase2_paired_benchmark.py) (Committed at git HEAD `e71f3b6`)
- **Generating Code Status**: Fully committed and tracked in git repository.
- **Dataset Lineage**: polyVERSE (`data/processed/processed.csv`, SHA256: `de47e0dca370ab7d48fec87ad0ce5fe80269c84ba7a0a0317a73732d53a7d1ac`)
- **Target Property & Units**: Polymer Bandgap ($E_g$) in **eV**
- **Evaluated Split**: `test` split (632 sealed test records from `data/processed/splits.json`)
- **Random Seed**: `20260815`
- **Variants per Attack ($n$)**: 5
- **Config / Manifest Snapshot**: Recorded in `reproducibility.json`

### Checkpoint & Scaler Hashes (SHA256)
| Component | Directory / Path | SHA256 Hash |
| :--- | :--- | :--- |
| **Clean Baseline Model** | `results/models/transformer_regressor/model.pt` | `0b1bb3f62ca72a897fd84c8e7f3129f8a7a9d75c1179c10debb63e46113731e4` |
| **Clean Target Scaler** | `results/models/transformer_regressor/scaler.json` | `5ef15ad16aee156fbc9b6b7872c1534262e3622022190f17c23fc7e3dab12814` |
| **Defended Model** | `results/models/transformer_defended/model.pt` | `bda109bd0bd7f7843c803ec31c842a5ed1190ed4714e1448e42c28d0c9e7c484` |
| **Defended Target Scaler**| `results/models/transformer_defended/scaler.json` | `5ef15ad16aee156fbc9b6b7872c1534262e3622022190f17c23fc7e3dab12814` |

*Note: The target scalers are 100% identical. Strict scaler freezing was maintained between baseline and defended models.*

---

## 2. Candidate Identity Pairing Verification

Row-by-row verification was executed across `attack_candidates.jsonl`, `clean_attack_records.jsonl`, and `defended_attack_records.jsonl`. 

For every single candidate index $i \in [0, 13862]$, the candidate ID, polymer ID, attack family, original representation, adversarial representation, and validity status match **100.0% exactly**.

| Attack Family | Baseline $N$ | Defended $N$ | Exact Matched $N$ | Candidate Identical? | Valid Representation $N$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`deletion`** | 3,102 | 3,102 | 3,102 | **True** | 1,544 (49.77%) |
| **`insertion`** | 3,160 | 3,160 | 3,160 | **True** | 1,387 (43.89%) |
| **`randomization`** | 3,157 | 3,157 | 3,157 | **True** | 3,082 (97.62%) |
| **`rearrangement`** | 1,363 | 1,363 | 1,363 | **True** | 773 (56.71%) |
| **`substitution`** | 3,081 | 3,081 | 3,081 | **True** | 1,604 (52.06%) |
| **TOTAL** | **13,863** | **13,863** | **13,863** | **True** | **8,390 (60.52%)** |

---

## 3. Eligibility Consistency & Population Audit

### Inconsistency Identified in `summary.json`
- `summary.json` computed `mean_abs_drift`, `median_abs_drift`, and `max_abs_drift` across **ALL** candidate rows (including invalid/unparseable strings).
- `summary.json` computed `success_rate` as $N_{\text{success}} / N_{\text{total}}$, using total candidates as the denominator rather than valid representation survivors.

### Independent Recomputation from Raw Records

#### Denominator A: Total Candidates ($N_{\text{total}}$) — As in `summary.json`
| Attack Family | Clean Mean Drift | Defended Mean Drift | Clean Succ % | Defended Succ % | Total-Denom Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`deletion`** | 0.3846 eV | 0.3821 eV | 15.47% | 14.41% | -1.06% |
| **`insertion`** | 0.4238 eV | 0.4235 eV | 14.11% | 14.15% | +0.03% |
| **`randomization`** | 0.6065 eV | 0.5907 eV | 47.39% | 47.48% | +0.10% |
| **`rearrangement`** | 0.1982 eV | 0.1377 eV | 5.36% | 1.69% | -3.67% |
| **`substitution`** | 0.2914 eV | 0.1551 eV | 9.51% | 3.12% | -6.39% |

#### Denominator B: Valid Representations Only ($N_{\text{valid}}$) — Scientifically Correct
| Attack Family | Clean Valid Drift (Mean/Med/Max) | Defended Valid Drift (Mean/Med/Max) | Clean Succ % | Defended Succ % | Valid-Denom Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`deletion`** | 0.3893 / 0.2780 / 3.4193 | 0.3944 / 0.2732 / 3.8125 | 31.09% | 28.95% | -2.14% |
| **`insertion`** | 0.4035 / 0.2800 / 3.5984 | 0.4426 / 0.2932 / 5.7103 | 32.16% | 32.23% | +0.07% |
| **`randomization`** | 0.6031 / 0.4431 / 4.7284 | 0.5889 / 0.4463 / 4.1483 | 48.54% | 48.64% | +0.10% |
| **`rearrangement`** | 0.1864 / 0.1147 / 2.5096 | 0.1179 / 0.0831 / 0.9881 | 9.44% | 2.98% | -6.47% |
| **`substitution`** | 0.2941 / 0.1492 / 3.3913 | 0.1509 / 0.0901 / 2.5275 | 18.27% | 5.99% | -12.28% |

---

## 4. Scientific Semantics & Test-Set Use

### Semantic Classification
- **Representation-Preserving Control**: `randomization` (SMILES re-ordering). Molecular graph is canonically identical; inherits measured bandgap label.
- **Chemistry-Changing Stress Tests**: `substitution`, `insertion`, `deletion`, `rearrangement`. These alter atomic composition or stoichiometry and are **NOT** label-preserving. In this evaluation script, they were scored as prediction drift stress tests relative to clean baseline predictions.

### Test-Set Pristine Status: CONTAMINATED
- `results/phase2_paired_benchmark/` was executed against the **632-sample sealed test split**.
- The sealed test set was repeatedly evaluated during earlier Phase 2 hyperparameter selection, scaler debugging, and model selection.
- **Verdict**: The sealed test set is no longer statistically pristine and must not be described as an un-inspected final test evaluation.

---

## 5. Discrepancy Summary Table

| Metric / Aspect | Recorded in `summary.json` / README | Recomputed from Raw Records | Cause of Discrepancy |
| :--- | :--- | :--- | :--- |
| **Substitution Success Rate** | 9.51% $\rightarrow$ 3.12% (Delta: -6.39%) | 18.27% $\rightarrow$ 5.99% (Delta: -12.28%) | `summary.json` used total candidates ($N=3081$) as denominator; raw valid denominator is $N=1604$. |
| **Rearrangement Success Rate** | 5.36% $\rightarrow$ 1.69% (Delta: -3.67%) | 9.44% $\rightarrow$ 2.98% (Delta: -6.47%) | `summary.json` used total candidates ($N=1363$) as denominator; raw valid denominator is $N=773$. |
| **Substitution Mean Drift** | 0.2914 eV $\rightarrow$ 0.1551 eV | 0.2941 eV $\rightarrow$ 0.1509 eV | `summary.json` included invalid representation rows in mean drift calculation. |
| **Test Set Integrity** | Described as "sealed test evaluation" | Evaluated multiple times in development | Test set inspected across Phase 2 iterations. |

---

## 6. Final Classification & Conclusion

### Classification: `VALID BUT DEVELOPMENTAL`

### Justification
1. **Why `VALID`**: Candidate identity pairing is 100% exact across all 13,863 rows, generating code is tracked in git, target scalers are perfectly frozen, and raw JSONL records are intact.
2. **Why `DEVELOPMENTAL`**:
   - The evaluated test set had been previously inspected during Phase 2 development.
   - `summary.json` aggregated mean/median drifts over invalid representations alongside valid ones.
   - The evaluated defended model (`transformer_defended`) was trained on atomic mutations that inherited measured bandgap targets.

### Can this replace the proposed new Phase 2 experiment?
**No**. While it confirms candidate bank pairing mechanics, it evaluates a defended model trained under naive label inheritance on a previously inspected test split.

### Do we need to rerun Phase 2?
**Yes**. A fresh, canonical Phase 2 experiment must be executed where defenses are trained with explicit label policies (SMILES-randomization defense and teacher-labeled stress test defense) and evaluated on clean validation/test candidate banks with bootstrap confidence intervals.
