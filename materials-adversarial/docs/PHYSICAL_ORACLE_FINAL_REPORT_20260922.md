
# SCIENTIFIC STATUS UPDATE (2026-09-22)
**Final Local Status:** CONVERGENCE_FAILED
**Reason:** Frozen convergence protocol (12 Å vacuum) exceeds the practical memory envelope of the local 15 GiB host (exhausted at 80/360 Ry on the 22-atom C/H smoke structure before SCF).

**Methodological State:**
- No protocol freeze occurred.
- No 12-polymer calibration was run.
- No attack-pair DFT was run.
- No DFT sign-agreement or magnitude-error results exist.
- No chemistry-changing drift is established as physical error.
- The finite capped-oligomer PBE HOMO-LUMO gap is not validated as the historical periodic-chain target.
- The 8 Å C/H smoke calculation is infrastructure evidence only, not calibration evidence.
- Frozen signed model predictions exist for four attack pairs (all negative) and remain unchanged.
- Test Suite: 462 passed, 1 skipped.
- Provisioning (QE 7.6 + SSSP 1.3.0) is verified.
- **Next defensible action:** Run the unchanged convergence design on a higher-memory host (after preregistering relaxation force threshold).

*(Note: Any prior statements regarding "QE unavailable" are superseded by this updated status.)*

---

# Physical-oracle study: convergence gate report

Date: 2026-09-22
Final status: **CONVERGENCE_FAILED**

## Motivation and evidence boundary

The intended study asks whether chemistry-changing model predictions agree in sign and magnitude with an independent finite-oligomer PBE calculation. Prediction drift is not physical error, and the capped-oligomer Kohn-Sham gap is not assumed to reproduce the historical periodic polyVERSE target. The sealed ML test split was not opened.

## Provisioning provenance

Quantum ESPRESSO 7.6, all required executable hashes, the SSSP 1.3.0 PBE Precision release, per-element UPF hashes, compilers, MPI, and linked numerical libraries are recorded under `results/physical_oracle_phase/20260921/provisioning/`. All three required executables launch, and the backend environment preflight passes.

## Pre-run audit

The 12 calibration identities, 30 successful structure hashes, six construction failures, four frozen attack-pair identities, and four signed frozen model predictions are deterministic and preserved. The attack-pair QE structures do not yet exist, appropriately, because pair construction is downstream of calibration.

Two critical protocol/execution problems remain:

1. The relaxation force threshold is explicitly unpinned in the prepared protocol.
2. The existing convergence plan starts at 12 angstrom vacuum on 56-, 86-, and 71-atom `n3` structures. The smaller 22-atom C/H smoke structure already exhausted practical local memory at 12 angstrom and 80/360 Ry before SCF.

See `results/physical_oracle_phase/20260922/pre_run_audit.json`.

## Convergence design and result

The intended sequential order—cutoff, vacuum, SCF threshold, relaxation, and repeat length—is scientifically preferable to a Cartesian sweep. The preregistered gap plateau criterion is at most 0.05 eV between successive levels, with exact electron count, converged SCF, no fatal warnings, and exact executable/UPF provenance.

No new point was launched because every frozen representative at the minimum 12 angstrom vacuum is strictly larger than a directly observed memory failure. Repeating a dominated configuration merely to crash would violate the resource constraint and add no scientific information.

Consequently, no cutoff, vacuum, SCF-threshold, relaxation, or repeat-length plateau was established. `oracle_protocol_frozen.json` and `oracle_protocol.sha256` were deliberately **not** created: naming an unconverged protocol “frozen” would be false provenance.

## Calibration and attack-pair results

Calibration was not started. Therefore MAE, RMSE, bias, correlations, bootstrap intervals, and a calibration verdict are unavailable—not zero. The four attack pairs were not constructed or executed. There are no DFT `delta_E`, sign-agreement, or magnitude-error results.

Empty schema-only CSV files and explicit `NOT_RUN_CONVERGENCE_FAILED` summaries preserve this distinction under `results/physical_oracle_phase/20260922/`.

## Supported claims

- The local pinned QE/SSSP environment can complete the 8 angstrom C/H infrastructure smoke.
- The existing 12 angstrom convergence floor is outside this host's practical envelope even for that smaller structure.
- Signed model deltas are recoverable and frozen, but remain model outputs only.
- The finite-oligomer oracle has not been calibrated and cannot adjudicate chemistry-changing attacks yet.

## Unsupported claims

- No chemistry-changing prediction has been shown to be a physical error.
- No model-versus-DFT sign or magnitude claim is supported.
- The smoke gap is not a calibration result.
- The finite capped-oligomer PBE gap is not established as a substitute for the historical periodic-chain target.

## Required next environment

Continue on a higher-memory host using the unchanged 12/15/18/22 angstrom vacuum plan, or explicitly preregister a newly versioned convergence protocol before further execution. Also lock a QE force threshold and relaxation failure rule before relaxation sensitivity work. Do not run calibration or attack pairs first.

## Reproducibility and artifacts

- Provisioning: `results/physical_oracle_phase/20260921/provisioning/`
- Smoke: `results/physical_oracle_phase/20260921/smoke/poly_405_capped_n2/`
- Audit: `results/physical_oracle_phase/20260922/pre_run_audit.json`
- Convergence outcome: `results/physical_oracle_phase/20260922/convergence_manifest.json`
- Calibration gate: `results/physical_oracle_phase/20260922/calibration_summary.json`
- Pair gate: `results/physical_oracle_phase/20260922/attack_pair_oracle_summary.json`
- Provenance index: `results/physical_oracle_phase/20260922/physical_oracle_provenance.json`

The last complete test run before this audit was 459 passed and 1 skipped. The suite is rerun after documentation and artifact validation.
