# Materials Adversarial Framework

**Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling**

This repository contains the complete experimental pipeline for assessing and mitigating adversarial vulnerabilities in Deep Learning models trained to predict physical polymer properties directly from 1D sequence representations.

## PROJECT
Deep Learning models applied to materials informatics often learn fragile syntactic shortcuts rather than true physicochemical representations. This project implements a unified adversarial framework that generates chemically valid perturbations to sequence representations, mathematically measures model vulnerability, and subsequently immunizes the network via targeted adversarial training.

## DATASET
- **Source**: polyVERSE (Ramprasad Group)
- **Target Property**: Bandgap (eV)
- **Usable Records**: 4,209 experimentally verified / high-fidelity DFT properties.
- **Split Configuration**: Stratified 80/10/10 split (Seed: `20260815`).
- *Note*: The `PI1M.csv` database acts as an unlabelled structure pool and is excluded via `.gitignore` to prevent repository bloat.

## BASELINE
- **Architecture**: Multi-head Transformer Regressor.
- **Input**: Tokenized PSMILES strings containing `[*]` attachment points.
- **Output**: Continuous prediction of Bandgap (eV).
- **Clean Performance**: Test MAE = `0.4619 eV`.

## ATTACKS
The framework utilizes sequence-level combinatorial generators constrained by RDKit valence parsing. An attack is defined as *successful* if it represents a chemically valid and plausible structure that causes absolute prediction drift exceeding the baseline Test MAE (`>0.4619 eV`).
- **Phase 1 Evaluation**: Over 11,800 valid candidate sequences were generated across the Validation Split. The baseline model was successfully manipulated by 4,232 sequences, exhibiting mean absolute prediction drifts of `~0.48 eV` and a maximum prediction drift of `5.47 eV`.

## DEFENSE (Adversarial Training)
To immunize the model without corrupting physical validity, an explicit **Label-Preservation Policy** was enforced. Only `Substitution` and `Rearrangement` attacks with a strict `attack_budget=1` were assumed to preserve the macroscopic Bandgap.
- `11,300` valid, label-preserving adversarial variants generated on the Train Split were appended to the baseline training pool. 
- The Defended Transformer was retrained from scratch, utilizing the frozen clean Target Scaler to guarantee mathematical comparability.

## RESULTS
The adversarially trained model reduced substitution attack success from **20.13%** to **7.53%** while clean Test MAE remained approximately unchanged (0.4619 → 0.4601 eV), but robustness did not transfer to the unseen insertion/deletion attacks evaluated.

| Evaluation Metric | Baseline Model | Defended Model |
| :--- | :--- | :--- |
| **Clean MAE** | 0.4619 eV | 0.4601 eV |
| **Substitution Success** | 20.13% | 7.53% |
| **Rearrangement Success** | 9.29% | 2.83% |
| **Insertion Success** | 32.61% | 32.46% |
| **Deletion Success** | 32.13% | 30.04% |

*(For comprehensive results and interpretation, view `docs/PHASE2_RESULTS.md`)*

## LIMITATIONS
- **Narrow Defense Generalization**: The defense did not demonstrate generalized robustness across the unseen attack families evaluated (Insertions/Deletions).
- **Heuristic Plausibility**: Representation validity (via RDKit parsing) is used as a proxy for structural plausibility. It is NOT proof of physical synthesis feasibility or true experimental behavior. 

## REPOSITORY STRUCTURE
- `configs/`: YAML configurations dictating datasets, architectures, and hyperparameters.
- `data/`: Raw downloaded CSVs, interim processed maps, and final deterministic splits.
- `docs/`: Comprehensive scientific documentation, Q&A, and pipeline logs.
- `results/`: Trained model binaries, scalers, and raw JSONL adversarial evaluation records.
- `scripts/`: Top-level executable scripts (`audit_dataset.py`, `train_baseline.py`, `eval_phase2_full.py`).
- `src/materials_adv/`: The core Python source code.
- `tests/`: 200+ PyTest integrity assertions validating string safety, deterministic splitting, and schema structures.

## HOW TO REPRODUCE
1. Environment Setup:
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[chem,dev]"
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```
2. Data Preprocessing:
```bash
PYTHONPATH=src .venv/bin/python src/materials_adv/data/preprocess.py
PYTHONPATH=src .venv/bin/python src/materials_adv/data/split_dataset.py
```
3. Read `docs/REPRODUCIBILITY.md` for exact test configuration seeds. All final Phase 2 metrics can be extracted automatically via `scripts/eval_phase2_full.py`.
