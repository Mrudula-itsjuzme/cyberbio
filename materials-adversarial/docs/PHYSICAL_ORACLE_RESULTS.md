# Physical-oracle phase results

Date: 2026-09-21
Outcome: **ONE INFRASTRUCTURE SMOKE COMPLETE; CONVERGENCE FAILED ON LOCAL HOST**

## Environment and reuse audit

- **MEASURED:** a pinned local build of Quantum ESPRESSO 7.6 was compiled from upstream commit `9f93ddec427d2b9a45bb72d828c6d324f62fcabd`. `pw.x`, `bands.x`, and `projwfc.x` each launch as version 7.6 and have recorded SHA-256 hashes.
- **MEASURED:** Open MPI 4.1.6 is installed. There are 12 logical CPUs and approximately 15 GiB RAM; no working NVIDIA driver was observed.
- **MEASURED:** SSSP 1.3.0 PBE Precision was pinned for Br, C, Cl, F, H, N, O, P, and S. Published MD5 values and local SHA-256 hashes are recorded in `results/physical_oracle_phase/20260921/provisioning/`.
- **MEASURED:** the QE backend now inserts atoms and cells, maps and hashes species/UPFs, checks electrons, emits isolated-cell SCF or optional relaxation inputs, extracts molecular band edges, and fails closed on incomplete or invalid output.
- **INTERPRETATION:** the local environment is sufficient for convergence work, but the finite-oligomer oracle is still uncalibrated.

## Prepared evidence

- 12 train/validation-only neutral calibration polymers.
- 36 attempted finite-oligomer structures across repeat counts 2, 3, and 4.
- 30 successful, hash-addressed SDF structures; six retained failures.
- A 30-job convergence matrix spanning low/mid/high target strata.
- A four-pair future pilot balanced across greedy/metropolis attacks and GraphMPNN/Transformer targets, with unique source molecules.
- Required elements: Br, C, Cl, F, H, N, O, P, S.
- Signed frozen-model predictions for the four future pairs, stored separately from oracle results. All four recomputed magnitudes match the frozen absolute drift; no oracle pair was run.

## Physical results

- **MEASURED INFRASTRUCTURE ONLY:** one C/H oligomer smoke SCF converged in 11 iterations at 80/360 Ry and 8 angstrom smoke-only vacuum. The parsed HOMO/LUMO values are -6.0614/-0.9402 eV and the finite orbital gap is 5.1212 eV.
- **BLOCKED:** zero calibrated structures and zero paired-oracle `delta_T`, `E_adv`, or `R` values exist.
- **BLOCKED:** no calibration quality classification can be made.
- **BLOCKED:** no attack direction has been confirmed, reversed, or weakened by an independent physical oracle.

The gap is not a scientific calibration result. It demonstrates executable, UPF, electron-count, SCF, and parser plumbing only.

## Compute accounting

- **MEASURED smoke:** 1 process x 22m31.99s = 0.3756 core-hours; peak RSS 7.484 GiB; raw scratch-inclusive storage 375 MiB. After recording the result, transient QE restart arrays were removed and 116 KiB of audit inputs, outputs, runtimes, and failure logs were retained.
- **PLANNING RANGE, NOT MEASURED:** 8-64 core-hours and 0.5-5 GB per structure were reserved only as a broad scheduling range pending one real smoke job.
- **PLANNING RANGE, NOT MEASURED:** evaluating all 20 pairs at three repeat counts would require 120 structures, corresponding to 960-7,680 core-hours and 60-600 GB under that provisional range.

These estimates are not suitability evidence and must be replaced with measured resource use before expansion.

## Decision

The 2026-09-22 pre-run audit stopped the study at **CONVERGENCE_FAILED**. The frozen convergence matrix begins at 12 angstrom vacuum on structures larger than the directly observed 12 angstrom memory failure, so no new dominated jobs were launched. Do not submit the 12-polymer set, any attack pair, or the sealed test split. Continue the unchanged matrix only on a higher-memory host, after locking the relaxation force threshold.
