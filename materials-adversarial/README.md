# Materials Adversarial Framework

**Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling**

This directory contains the core experimental pipeline for evaluating and improving the adversarial robustness of sequence-based Deep Learning models predicting polymer physical properties directly from 1D representations (PSMILES).

> [!NOTE]
> **Lineage Context**: Active research focuses exclusively on **polyVERSE Bandgap prediction ($E_g$ in eV)**. Earlier exploratory work on polymer glass-transition temperature ($T_g$ in Kelvin) is documented separately in [`docs/HISTORICAL_TG_LINEAGE.md`](docs/HISTORICAL_TG_LINEAGE.md) and must not be mixed with active configs or evaluation.

---

## ACTIVE PROJECT SPECIFICATION

### DATASET
- **Source**: polyVERSE (Ramprasad Group)
- **Target Property**: Bandgap ($E_g$ in eV)
- **Usable Records**: 4,209 high-fidelity DFT polymer property records.
- **Split Configuration**: Deterministic random 70/15/15 split (Seed: `20260815`), materialized in `data/processed/splits.json`:
  - **Train**: 2,946 records
  - **Validation**: 631 records
  - **Test (Sealed)**: 632 records (`test_sealed: true`)

### BASELINE MODEL
- **Architecture**: Multi-head Transformer Encoder Regressor.
- **Input**: Tokenized PSMILES strings containing `[*]` polymer attachment points.
- **Output**: Continuous bandgap prediction in eV.
- **Verified Clean Performance**: Test MAE = **0.4619 eV**, Test RMSE = **0.6235 eV**, Test $R^2$ = **0.8019** (Checkpoint: `results/models/transformer_regressor/`).

---

## ATTACK SEMANTICS & CLASSIFICATION

Adversarial sequence modifications are evaluated under strict scientific semantic boundaries:

1. **Representation-Preserving Control**:
   - **RDKit SMILES Randomization**: Generates non-canonical SMILES strings for the *identical* molecular graph. This represents the only provably label-preserving control.
2. **Chemistry-Changing Stress Tests**:
   - **Substitution, Insertion, Deletion, Rearrangement, Metropolis-Style Stochastic Search**: Edit atomic tokens or sequence structures. These operations alter chemical identity or stoichiometry and are **NOT** label-preserving. For defense training, their label basis is explicitly clean-model teacher predictions or excluded from supervised target inheritance.

An attack candidate is defined as *successful* if it passes chemical validity parsing and induces absolute prediction drift exceeding the baseline test MAE ($>0.4619\text{ eV}$).

---

## VERIFIED PAIRED BENCHMARK RESULTS

Trained baseline and defended models were evaluated on an identical, frozen bank of 13,863 attack candidates across the 632-sample sealed test set (`results/phase2_paired_benchmark/summary.json`):

| Evaluation Row | Attack Category | Total Candidates | Valid Candidates | Baseline Success Rate | Defended Success Rate | Paired Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SMILES Randomization** | Representation Control | 3,157 | 3,082 | 47.39% | 47.48% | +0.10% |
| **Substitution ($b=1$)** | Chemistry Stress Test | 3,081 | 1,604 | 9.51% | 3.12% | -6.39% |
| **Rearrangement ($b=1$)** | Chemistry Stress Test | 1,363 | 773 | 5.36% | 1.69% | -3.67% |
| **Insertion ($b=1$)** | Chemistry Stress Test | 3,160 | 1,387 | 14.11% | 14.15% | +0.03% |
| **Deletion ($b=1$)** | Chemistry Stress Test | 3,102 | 1,544 | 15.47% | 14.41% | -1.06% |

### Key Findings
- **Targeted Defense Reduction**: Adversarial training on substitution/rearrangement stress tests reduced substitution success rate from 9.51% to 3.12% and rearrangement success rate from 5.36% to 1.69%.
- **Zero Transfer to Unseen Stress Tests**: The defense provided zero protection against insertion (14.11% $\rightarrow$ 14.15%) or deletion (15.47% $\rightarrow$ 14.41%) stress tests.
- **Invariance Unsolved**: SMILES randomization drift remained high (~47.4% success rate across both models), confirming that string representation sensitivity is unmitigated by localized token defense.

---

## SCIENTIFIC LIMITATIONS
- **Defense Generalization**: Localized token defense does not generalize to unseen mutation types or representation-level SMILES shifts.
- **Validity vs Physicality**: RDKit validity parsing verifies representation syntax, not thermodynamic stability or synthesis feasibility.
- **Label Inheritance**: Chemistry-altering edits must be treated as stress tests; they do not preserve true experimental bandgap labels.

---

## REPOSITORY STRUCTURE
- `configs/`: Active YAML configurations (`dataset.yaml`, `model.yaml`, `tokenizer.yaml`, `attack.yaml`).
- `data/`: Raw CSV inputs, preprocessed outputs, and deterministic split definitions (`splits.json`).
- `docs/`: Canonical project state (`CANONICAL_PROJECT_STATE.md`), historical lineage documentation (`HISTORICAL_TG_LINEAGE.md`), and scientific audits.
- `results/`: Verified model checkpoints, scalers, and reproducible evaluation summary artifacts.
- `scripts/`: Executable pipeline scripts (`run_phase2_paired_benchmark.py`, `run_attack_efficiency.py`, `run_closed_loop.py`, `run_representation_attribution.py`).
- `src/materials_adv/`: Core Python library modules.
- `tests/`: 250 unit tests asserting model integrity, attack safety, and deterministic pipeline behavior.

---

## HOW TO RUN ACTIVE PIPELINE
1. **Environment Setup**:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[chem,dev]"
   ```
2. **Run Test Suite**:
   ```bash
   PYTHONPATH=src .venv/bin/python -m pytest -p no:launch_testing
   ```
3. **Canonical Paired Evaluation**:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_phase2_paired_benchmark.py
   ```
