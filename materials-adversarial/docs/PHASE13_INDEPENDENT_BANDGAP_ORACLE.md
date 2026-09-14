> [!NOTE]
> **Status**: CANONICAL

# Phase 13: Independent Bandgap Oracle Design

## Overview
Phase 13 establishes the required properties of an independent ground-truth property oracle to evaluate the adversarial chemistry-changing attacks developed in Phase 12B. This phase serves as a formal feasibility study since active DFT compute is unavailable in the automated environment.

## 1. Target Provenance & Limitations (RQ1)
The dataset utilized is **polyVERSE**, curated by the Ramprasad Group. The exact source file is `bandgap_chain.csv`.
- **Type**: COMPUTATIONAL_DFT
- **Method**: UNKNOWN (Specific functional and basis set are undocumented in the local file)
- **Protocol**: 1D periodic polymer chain calculation
- **Units**: eV

*Crucial Limitation*: The exact historical functional configuration was not persisted alongside the `bandgap_chain.csv` target values. Any future matched oracle must experimentally calibrate itself against the dataset (via `calibration_manifest.csv`) to ensure an exact property match before evaluating adversarial candidates.

## 2. Oracle Requirements (RQ2-RQ5)
An independent oracle must compute the identical physical property without relying on the attacked model's internal representations or weights.

### A. Matched Oracle (Recommended)
- **Definition**: A 1D periodic DFT calculation designed to match the original dataset. Since the exact DFT functional and basis set are UNKNOWN, the matched oracle specification cannot yet be finalized.
- **Polymer Representation Strategy**: The interpretation of `[*]` wildcard tokens depends entirely on the target provenance. Potential strategies include:
  - **Periodic Reconstruction**: Converting wildcards to periodic boundary conditions (if the target is periodic bandgap).
  - **Oligomer Construction**: Creating multi-unit chains.
  - **Capped Repeat Unit**: Capping with Hydrogen/Methyl groups.
- **Validation (RQ5)**: The oracle must evaluate standard metrics (MAE, RMSE, Bias, Pearson) on the 20-sample `calibration_manifest.csv` and justify acceptability scientifically without arbitrary thresholds.

### B. Calibratable & Mismatched Oracles
- Mismatched oracles (like calculating the HOMO-LUMO gap of a hydrogen-capped oligomer trimer) can yield systematic offsets from periodic bandgaps and should **not** be used as raw ground truth without a learned regression calibrator.

## 3. True Adversarial Error Definition (RQ7)
Raw model prediction drift ($\Delta_G = G_{adv} - G_{src}$) cannot be treated as error because the true chemistry changed. The actual error in predicting the physical perturbation must be defined relative to the oracle change ($\Delta_T = T_{adv} - T_{src}$):

$$ E_G = |\Delta_G - \Delta_T| $$

Only when $E_G$ is large can we scientifically conclude the model suffered an adversarial failure (either oversensitive or undersensitive).

## 4. Phase 12B Candidate Triage (RQ6)
A shortlist of 20 unique deterministic adversarial pairs (`candidate_shortlist_verified.csv`) was generated from Phase 12B trajectories. These include:
- Candidates yielding high drift exclusively on GraphMPNN.
- Candidates yielding high drift exclusively on Transformer.
- Candidates with moderate drift across both.
- Varying perturbation budgets (1, 2, and 3 token edits).
- Complete evaluations across both models are numerically verified.

## 5. Feasibility Verdict (RQ8)
Full oracle-labelled adversarial evaluation is **currently not feasible** without attaching a high-throughput DFT backend (like VASP or Quantum Espresso) to the evaluation pipeline. However, the exact experimental protocol, candidate list, calibration set, and metric definitions are now fully formalized and persisted in `results/phase13a_provenance/run_1/`.

## Conclusion
Chemistry-changing drift represents the model's sensitivity to structural edits, not its absolute error. While the GraphMPNN remains the canonical architecture due to its perfect equivalent-SMILES invariance and structural efficiency, we **cannot** definitively label its maximum drift as a failure without DFT supervision. Until an oracle is deployed, true attack $\to$ defend training for chemistry-changing edits remains not scientifically justified under the current evidence and compute environment.
