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

## 8. Final Experiment Execution: 5-Seed Paired Robustness Benchmark

*Note: The originally proposed Phase 0 paired benchmark has been fully superseded and implemented as the Canonical 5-Seed Benchmark.*

### Objective Completed
A rigorous, model-independent paired robustness benchmark comparing clean, representation-augmented, and MCMC-defended models on identical candidate banks has been successfully executed across 5 independent random seeds. 

**Conclusion & Final State**:
The project is considered analytically complete. The remaining work focuses exclusively on narrative framing (acknowledging representation sensitivity as an architectural finding rather than a bug) and defining the requirements for an independent physical oracle (e.g., DFT) as future work to validate the ground-truth properties of chemistry-changing adversarial candidates.
