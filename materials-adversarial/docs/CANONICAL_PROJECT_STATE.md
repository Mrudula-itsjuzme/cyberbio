# Canonical Project State Document

**Project**: Materials Adversarial Framework (`materials-adversarial`)  
**Repository**: `Mrudula-itsjuzme/cyberbio`  
**Current Branch**: `main`  
**Last Audit**: September 12, 2026

---

## 1. Lineage Classification

### Active Research Lineage (polyVERSE Bandgap)
- **Target Property**: Polymer Bandgap ($E_g$).
- **Units**: eV.
- **Usable Records**: 4,209 high-fidelity DFT polymer property records.
- **Split Configuration**: Deterministic random 70/15/15 split (Seed: `20260815`), materialized in `data/processed/splits.json`:
  - Train: 2,946 records
  - Validation: 631 records
  - Test (Sealed): 632 records (`test_sealed: true`)
- **Status**: Sole active research focus. All current configs, evaluation scripts, and models belong to this lineage.

### Historical Exploratory Lineage (OpenPoly $T_g$)
- **Target Property**: Polymer Glass-Transition Temperature ($T_g$).
- **Units**: Kelvin (K).
- **Usable Records**: ~247 records (scaffold split: 204 train / 37 val / 6 test).
- **Status**: Archived for historical research context in [`docs/HISTORICAL_TG_LINEAGE.md`](HISTORICAL_TG_LINEAGE.md). Must NOT be mixed into active configs or evaluation pipelines.

---

## 2. Baseline Model & Verified Metrics

- **Architecture**: Multi-head Transformer Encoder Regressor.
- **Input**: Tokenized PSMILES strings containing `[*]` attachment points.
- **Output**: Continuous prediction of Bandgap ($E_g$ in eV).
- **Verified Clean Performance** (`results/models/transformer_regressor/metrics.json`):
  - **Test MAE**: `0.4619 eV`
  - **Test RMSE**: `0.6235 eV`
  - **Test $R^2$**: `0.8019`
  - **Training Set Size**: 2,946 records
  - **Validation Set Size**: 631 records
  - **Test Set Size**: 632 records (Sealed)

---

## 3. Attack & Defense Inventory

### Attack Inventory
1. **SMILES Randomization (`randomization`)**: Generates non-canonical SMILES strings for the identical molecular graph. *Representation-Preserving Control*.
2. **Substitution (`substitution`)**: Single-token atomic substitution ($b=1$). *Chemistry-Changing Stress Test*.
3. **Insertion (`insertion`)**: Single-token atomic insertion ($b=1$). *Chemistry-Changing Stress Test*.
4. **Deletion (`deletion`)**: Single-token atomic deletion ($b=1$). *Chemistry-Changing Stress Test*.
5. **Rearrangement (`rearrangement`)**: Local token position swap ($b=1$). *Chemistry-Changing Stress Test*.
6. **Metropolis-Style Stochastic Search (`probabilistic_mcmc`)**: Multi-step model-guided candidate proposal with Metropolis acceptance filter. *Chemistry-Changing Stress Test*.

### Defense Inventory
1. **Baseline Clean Model (`transformer_regressor`)**: Standard MSE training on clean train split (2,946 samples).
2. **Adversarially Defended Model (`transformer_defended`)**: Trained with clean dataset augmented by substitution and rearrangement variants ($b=1$) under a strictly frozen clean target scaler.

---

## 4. Semantic Label Classification

- **A. Representation-Preserving Control**:
  - **SMILES Randomization**: Preserves canonical molecular graph topology. Inherits measured physical target ($E_g$) because chemical identity is invariant under SMILES re-ordering.
- **B. Chemistry-Changing Stress Tests**:
  - **Substitution, Insertion, Deletion, Rearrangement, Metropolis-Style Search**: These edits modify atomic composition, stoichiometry, or molecular bonding. They are **NOT** label-preserving. For defense training, their target labels must be clean-model teacher predictions or excluded from supervised target inheritance.

---

## 5. MCMC Terminology Audit Verdict

- **Audit Finding**: The `probabilistic_mcmc` implementation proposes edits using deterministic sub-attacks and filters candidates using a Metropolis ratio based on model prediction drift.
- **Verdict**: It is **NOT** a formal Metropolis-Hastings MCMC sampler because it does not track asymmetric proposal ratios $q(x|y)/q(y|x)$ or evaluate stationary distribution convergence.
- **Action Taken**: Formally documented and renamed in code/docs as **Metropolis-style model-guided stochastic search**.

---

## 6. Verified Paired Benchmark Results

Evaluated on an identical, frozen bank of 13,863 candidate sequences across the 632-sample sealed test split (`results/phase2_paired_benchmark/summary.json`):

| Evaluation Row | Category | Total Candidates | Valid Candidates | Baseline Success | Defended Success | Paired Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SMILES Randomization** | Representation Control | 3,157 | 3,082 | 47.39% | 47.48% | +0.10% |
| **Substitution ($b=1$)** | Chemistry Stress Test | 3,081 | 1,604 | 9.51% | 3.12% | -6.39% |
| **Rearrangement ($b=1$)** | Chemistry Stress Test | 1,363 | 773 | 5.36% | 1.69% | -3.67% |
| **Insertion ($b=1$)** | Chemistry Stress Test | 3,160 | 1,387 | 14.11% | 14.15% | +0.03% |
| **Deletion ($b=1$)** | Chemistry Stress Test | 3,102 | 1,544 | 15.47% | 14.41% | -1.06% |

### Trusted Artifacts
- `results/models/transformer_regressor/metrics.json`
- `results/phase2_paired_benchmark/summary.json`
- `results/phase2_paired_benchmark/paired_robustness.json`
- `data/processed/splits.json`

### Untrusted / Superseded Claims & Artifacts
- **Superseded**: Older README table numbers (e.g., 20.13% substitution success) derived from un-paired or validation-set runs.
- **Refuted**: The historical hypothesis that length-changing attacks are uniquely dangerous (refuted in Phase 2F).
- **Refuted**: The claim that token substitution/rearrangement attacks are label-preserving.

---

## 7. Current Scientific Limitations

1. **Local Defense Generalization**: Adversarial training on seen mutation types ($b=1$ substitution/rearrangement) reduces drift on those specific mutation types but provides zero transfer to unseen insertion/deletion attacks.
2. **Representation Sensitivity**: High success rate under SMILES randomization (~47.4%) shows that sequence models remain sensitive to representation re-ordering even when physical chemistry is invariant.
3. **Syntax vs Physicality**: RDKit validity verifies syntax parsing, not synthesizability or thermodynamic stability.

---

## 8. Multi-Seed Experiment Suites

Two separate multi-seed experiment suites are retained and must be reported separately:

1. **Defense-comparison suite (3 seeds)**: `42`, `123`, `456`, produced by `scripts/run_multiseed.py`. This suite compares Baseline, RandSmilesAug, MCMCDefended, and CombinedDefense configurations across RandomSplit and ScaffoldSplit settings.
2. **Comprehensive robustness benchmark (5 seeds)**: `42`, `123`, `2026`, `777`, `999`, produced by the comprehensive benchmark workflow and reported in the corresponding benchmark artifacts.

These suites answer related but different questions and must not be pooled into a single seed count or treated as interchangeable evidence.

**Conclusion & Final State**:
The core analytical framework is mature, but claims should remain tied to the exact artifact and experiment suite that produced them. Representation sensitivity should be framed as an observed model behavior, and chemistry-changing perturbations still require an independent physical oracle (for example, DFT) before claims about true property preservation can be made.


---

## 9. September 21 Scientific Review — Canonical Addendum

This addendum supersedes older repository-wide test-status and representation-inference statements.

### 9.1 Current test status

The current scientific-review branch reports:

- **448 passed**
- **1 skipped**
- **1 warning**

The older `401 passed, 17 failed, 11 errors` snapshot and older `429 passed` claims describe earlier repository states. They are historical audit evidence, not the current canonical test status.

### 9.2 Corrected label schema

New chemistry-changing candidate banks use:

- `source_target`
- `candidate_target: null`
- `label_basis: candidate_target_requires_independent_physical_oracle`

Equivalent-SMILES records may inherit the source property only after graph-equivalence validation.

### 9.3 Validation-only multi-view inference result

The 631-molecule validation split was evaluated with up to 16 deterministic equivalent-SMILES views per molecule, with 15.97 unique views obtained on average. No sealed test sample was opened and no model was retrained.

| Model | Canonical validation MAE | Mean-ensemble validation MAE | ΔMAE |
|---|---:|---:|---:|
| Ordinary Transformer | 0.5186 eV | 0.5604 eV | +0.0418 eV |
| Augmented Transformer | 0.5009 eV | 0.4844 eV | -0.0165 eV |
| GraphMPNN | 0.4112 eV | 0.4112 eV | ≈0 |

Paired molecule-bootstrap intervals for ΔMAE:

- Ordinary Transformer: **+0.0022 to +0.0832 eV**
- Augmented Transformer: **-0.0274 to -0.0060 eV**
- GraphMPNN: approximately **-5.9e-9 to +1.0e-8 eV**

The strongest supported negative result is that **multi-view inference does not rescue the frozen ordinary Transformer**.

### 9.4 Representation disagreement

For the ordinary Transformer, equivalent-view prediction standard deviation is associated with absolute error:

- Pearson: **0.291**
- Spearman: **0.241**
- low-disagreement quartile mean error: **0.474 eV**
- high-disagreement quartile mean error: **0.774 eV**
- error >0.5 eV: **34.8%** versus **61.4%**

This is a diagnostic association, not a calibrated uncertainty claim and not causal evidence.

### 9.5 Remaining publication limitation

Existing augmented-Transformer and GraphMPNN evaluation is reproducible under the new protocol, but their original training is not fully publication-reproducible because complete deterministic training provenance was not preserved. Only one trained seed per architecture is evaluated here.

The next publication-grade ML experiment is frozen multi-seed retraining with complete checkpoint/tokenizer/scaler/split/environment provenance, followed by the same validation protocol and exactly one final sealed-test evaluation after all choices are frozen.

Detailed source: [REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md](REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md).
