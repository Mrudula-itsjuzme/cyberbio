# Materials Adversarial Robustness Framework

This repository provides a framework for evaluating the adversarial robustness of machine learning models predicting materials properties (e.g., electronic bandgap of polymers) from SMILES strings or molecular graphs.

## Core Findings (Canonical)
- Sequence-to-sequence models (Transformers) exhibit extreme sensitivity to equivalent-SMILES representations.
- Graph Neural Networks (GraphMPNN) structurally eliminate representation-based equivalent-SMILES vulnerabilities while retaining predictive accuracy (MAE ~0.411 eV).
- Chemistry-changing adversarial attacks induce prediction drift, evaluated as follows:
  * equivalent-SMILES: CANONICAL
  * substitution: CANONICAL
  * deletion: CANONICAL
  * bounded multi-substitution: CANONICAL
  * insertion: NOT TESTED / future extension

## HPC Oracle Migration
True attack $\rightarrow$ defend adversarial training against chemistry-changing edits requires an independent physical oracle (e.g., Density Functional Theory) to calculate the true ground-truth property of the newly generated adversarial structure.

**Status**: HPC oracle execution framework scaffold. Oracle-compatible quantum structure construction/execution is unavailable in the current environment.
A backend-agnostic HPC Oracle Migration package has been prepared in the `hpc_oracle/` directory. Deploy this package to a SLURM-managed cluster with Quantum ESPRESSO or VASP to compute the true adversarial error ($E_S$ and $E_G$).

## Limitations
Currently, chemistry-changing prediction drift represents *model sensitivity*, NOT confirmed absolute error. Adversarial retraining against these chemistry changes without an independent physical oracle is not scientifically justified under the current evidence and compute environment.

## Documentation
Please see the `docs/` folder for the complete project history.
- `FINAL_RESULTS.md`: Canonical metrics and maximum observed drifts.
- `PROJECT_STORY.md`: High-level narrative of the scientific discovery process.
- `CLAIMS_LEDGER.md`: Ledger of all scientific hypotheses tested and their resolutions.
