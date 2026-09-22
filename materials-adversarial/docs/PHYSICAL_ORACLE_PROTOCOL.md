# Independent physical-oracle protocol

Date locked: 2026-09-21
Status: **PROVISIONED; ONE SMOKE COMPLETE; CONVERGENCE FAILED ON LOCAL HOST**

## Claim boundary

- **MEASURED:** pinned QE 7.6 executables and SSSP 1.3.0 PBE Precision files for all required elements are locally installed and hash-verified. See `results/physical_oracle_phase/20260921/provisioning/`.
- **MEASURED:** the QE backend and fail-closed gap parser passed focused tests and one real molecular SCF smoke. Prior `FINISHED` markers remain process-status artifacts, not physical results.
- **INTERPRETATION:** PSMILES supplies repeat connectivity but not a unique repeat translation, torsional state, packing motif, or periodic cell. A defensible periodic-chain structure cannot be reconstructed from the available records without inventing geometry.
- **HYPOTHESIS:** a consistently computed, hydrogen-terminated finite-oligomer PBE HOMO-LUMO gap may be useful as a calibrated estimator of paired polymer-chain gap changes.
- **BLOCKED:** no claim is made that this surrogate reproduces the historical polyVERSE target-generation pipeline. Its functional, pseudopotentials, geometry, and convergence settings are not available in the repository.

## Quantity and discrepancy definitions

For source `s` and candidate `c`, using the same repeat count and converged oracle protocol:

`delta_T = E_oracle(c) - E_oracle(s)`

The corresponding signed model change must be recovered as:

`delta_G = G_model(c) - G_model(s)`

Report:

- `E_adv = abs(delta_G - delta_T)`
- `R = delta_G - delta_T`

Absolute model drift is insufficient to compute either signed diagnostic. Signed predictions have now been recovered in `model_predictions/signed_model_predictions.csv`; the paired pilot remains blocked on convergence and calibration.

## Frozen surrogate construction

1. Accept exactly two terminal wildcard atoms, each with degree one and matching attachment-bond type.
2. Reject non-neutral, radical, connectivity-ambiguous, or chemically suspect candidates. In particular, parser-induced hydrogenated halogens and hypervalent hydrogenated sulfur are rejected.
3. Build repeat-count 2, 3, and 4 linear oligomers from a fixed head/tail convention.
4. Remove terminal wildcards and allow implicit hydrogen capping. This is a finite molecular surrogate, not a periodic polymer.
5. Generate conformers with ETKDGv3 and the fixed seed in `hpc_oracle/configs/surrogate_structure_protocol.yaml`.
6. Pre-optimize with MMFF94 when parameterized, otherwise UFF; record the fallback. Reject force-field non-convergence instead of retaining it.
7. Preserve SDF hashes, coordinate spans, charge, multiplicity, elements, and construction failures in the structure manifest.

No candidate in the attack shortlist has yet been converted to a DFT input or evaluated.

## Frozen electronic-structure policy

- Code: Quantum ESPRESSO, exact version to be recorded at execution.
- Functional: PBE as a prospective protocol choice, not a recovered historical setting.
- Pseudopotentials: the single curated SSSP 1.3.0 PBE Precision release across all required elements. SSSP deliberately selects different generator/types by element; no file may be substituted from another SSSP release or arbitrary source.
- Wavefunction and charge-density cutoffs: derive the baseline from the selected UPF recommendations. Test 1.00, 1.25, and 1.50 times the maximum recommended wavefunction cutoff and converge the density cutoff consistently. Do not reuse the repository's unvalidated 40/320 Ry defaults.
- Boundary condition: isolated oligomer in a three-dimensional supercell, Gamma only.
- Vacuum study: 12, 15, 18, and 22 angstrom in every non-molecular direction after accounting for molecular span.
- Occupations: fixed, closed-shell only after electron-count validation. Smearing requires a separately documented ambiguity study.
- SCF thresholds: `conv_thr` of 1e-6, 1e-8, and 1e-10 Ry.
- Geometry: run a QE relaxation in the selected cell. The force threshold must be locked, in QE units, before the first execution.
- Gap extraction: parse verified occupied and unoccupied eigenvalues. Fail closed on SCF failure, incomplete relaxation, ambiguous occupation, or metallic result; never coerce such output into a gap.

Because this is an isolated molecular surrogate, a k-point sweep would be scientifically meaningless. A future periodic-chain oracle would require a new construction protocol and an explicit one-dimensional k-point convergence study.

## Staged execution gate

1. Install a pinned QE release and verified SSSP files for every element in `required_elements.json`.
2. Replace every placeholder in `required_pseudopotentials.csv` and record hashes and recommended cutoffs.
3. Implement and unit-test complete QE input generation, relaxation/SCF state handling, and gap parsing.
4. **COMPLETE:** one representative smoke calculation replaced planning ranges with measured time and storage. Its 8 angstrom vacuum is smoke-only and does not alter the frozen convergence series.
5. Execute the low/mid/high convergence plan. Freeze the cheapest settings that produce stable absolute and paired gaps; report the full curves rather than selecting a threshold post hoc.
6. Execute only the train/validation calibration set. The sealed ML test split remains untouched.
7. Classify the surrogate from observed calibration and length-dependence evidence.
8. Only if the calibration supports the intended paired interpretation, construct and run the four-pair pilot.
9. Do not automatically expand to the full 20-pair shortlist.

# PREREGISTERED RELAXATION CRITERION

This document explicitly preregisters the geometry-relaxation criteria for the PBE finite oligomer study prior to execution on a higher-memory host. This is a preregistered design choice, not a result-derived setting. It must not be tuned from future attack-pair outcomes.

## Criteria
- **Force Threshold (`forc_conv_thr`)**: 1.0D-3 Ry/Bohr
- **Total Energy Threshold (`etot_conv_thr`)**: 1.0D-4 Ry
- **Stopping Condition**: The structural relaxation is considered converged when all components of all forces are less than the force threshold AND the change in total energy between successive steps is less than the energy threshold.
- **Action on Failure**: If the maximum number of steps (`nstep=100`) is reached without convergence, the structure is marked as `FAILED_RELAXATION`. No fallback to looser criteria is permitted.

## Rationale
These values represent a conservative standard for Quantum ESPRESSO PBE calculations on finite oligomers, ensuring precise forces for reliable HOMO-LUMO gap extraction without imposing impossible tolerances for soft polymeric degrees of freedom.
