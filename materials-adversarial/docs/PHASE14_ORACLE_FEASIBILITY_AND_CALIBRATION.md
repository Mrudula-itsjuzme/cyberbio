> [!WARNING]
> **Historical Document / Superseded Methodology**
> This file chronicles an earlier development phase of the project (Phases 1-14). Early iterations of the MCMC defense in these logs evaluated closed-loop training using **original-label inheritance** for chemistry-changing edits. As established in the Phase 9 Scientific Audit, this assumption is physically invalid without a DFT oracle.
> The final canonical benchmark (`canonical_benchmark_no_leakage.json` and the final `EXPERIMENTAL_PROTOCOL.md`) explicitly abandons label inheritance in favor of **label-free consistency regularization**, and promotes **Rand-SMILES Augmentation** as the primary, physically sound defense. Please refer to `RESEARCH_CONTRIBUTIONS.md` for the current scientific consensus.

> [!NOTE]
> **Status**: BLOCKED

# Phase 14: Independent Oracle Feasibility and Calibration Pilot

## 1. Verified Target Semantics (RQ1)
Based on Phase 13A provenance tracing, the canonical target definition is:
- **Dataset Origin**: Ramprasad Group `polyVERSE`
- **Source File**: `Other/bandgap_chain.csv`
- **Property**: Electronic Bandgap
- **Structural Representation**: 1D periodic polymer chains
- **Method**: Computational DFT
- **Units**: eV
- **DFT Functional**: UNKNOWN
- **Basis Set / Pseudopotential**: UNKNOWN

Since the exact historical DFT functional configuration is UNKNOWN, any oracle match claim requires empirical calibration. No arbitrary universal oracle acceptance threshold can be predefined.

## 2. Available Oracle Tools (Local Environment Audit)
An audit of the automated environment was conducted to identify practical oracle options for executing a small calibration pilot. The following widely used quantum chemistry and electronic structure tools were queried via system PATH and Python environment (`pip list`):

- **Quantum ESPRESSO**: NOT INSTALLED
- **GPAW**: NOT INSTALLED
- **CP2K**: NOT INSTALLED
- **PySCF**: NOT INSTALLED
- **xTB / GFN2-xTB**: NOT INSTALLED
- **ASE (Atomic Simulation Environment)**: NOT INSTALLED

Because large external packages cannot be installed blindly, we currently have **zero** local software options capable of running either 1D periodic DFT or tight-binding calculations.

## 3. Polymer Construction Feasibility
Before running any calculation, the representation of `[*]` wildcards must be structurally resolved.
If tools were available, building 1D periodic boundary conditions (PBC) would be the strictly correct provenance-based approach. Without a periodic-capable DFT code, falling back to oligomer capping (e.g., hydrogen-capped trimers) would introduce a known physical mismatch (HOMO-LUMO gap of an oligomer vs. true bulk bandgap).

However, without any QC software to parse SMILES into 3D geometries or unit cells, all construction strategies currently fail at the "software limitation" stage.

## 4. Oracle Classifications
If the software were present, we would classify them as follows:
- **MATCHED (Class A)**: A 1D periodic DFT calculation calibrated to the dataset.
- **CALIBRATABLE (Class B)**: E.g., PySCF oligomer calculations or xTB tight-binding, which calculate a mismatched property requiring learned regression.
- **MISMATCHED (Class C)**: Properties completely orthogonal to electronic excitation (e.g., purely steric or mechanical properties).

Given the environment, all options are effectively mapped to `NOT_FEASIBLE_IN_CURRENT_ENVIRONMENT`.

## 5. Calibration Pilot Results
A deterministic calibration set (`calibration_manifest_verified.csv`, N=20 samples) spanning diverse structural motifs and bandgap ranges (low/medium/high) was staged.

**Outcome**: The calculations were **NOT RUN**.

### Failure Modes
Every calibration sample failed immediately.
- **Failure Category**: Software limitation
- **Failure Reason**: No quantum chemistry software is installed to generate 3D geometries or execute energy calculations. 

## 6. Oracle Validation and Delta-Based Validation
Because zero calibration calculations succeeded, we cannot compute MAE, R², or Pearson correlations. It remains unknown whether systematic offsets exist or if Delta_oracle tracking is scientifically defensible.

## 7. Decision Rule (Oracle Usability Verdict)
**Verdict**: `NOT_FEASIBLE_IN_CURRENT_ENVIRONMENT`

We cannot evaluate the 5-pair adversarial pilot because we lack the basic computational engine required to derive `T_source` and `T_candidate`.

## 8. Conclusion and Implications
While the Phase 12B adversarial search successfully generated valid chemistry-changing structures causing up to ~3.19 eV GraphMPNN drift, we **cannot** classify these candidates as true adversarial failures today. Model drift does not automatically imply model error when the underlying physics changes. 

True attack $\to$ defend adversarial training remains scientifically blocked.

**Recommended Next Step:**
Do not return to adversarial generation (GAN/LLM) or retraining on blind candidate drift. The project should either conclude its current scope (publishing the framework, constraints, and canonical invariance) or be deployed to an environment with a dedicated HPC backend (e.g., a slurm cluster with VASP or Quantum ESPRESSO) to execute the fully defined Phase 14 calibration and pilot tests.
