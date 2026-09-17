# Surrogate QC Calibration Pilot

This document defines the physical calibration exercise for the `CALIBRATABLE_SURROGATE` 3D geometries. Because the local environment currently lacks Quantum ESPRESSO (`pw.x`) and VASP binaries, this pilot acts as a fully constructed dry-run. Its execution outcome is formally logged as `BACKEND_EXECUTION_BLOCKED`.

## Calibration Questions

When executed, the surrogate QC runs must answer the following:
1. **Q1:** Does surrogate-QC bandgap correlate with dataset bandgap?
2. **Q2:** Is there systematic bias?
3. **Q3:** Does rank ordering survive?
4. **Q4:** How sensitive is the QC result to oligomer length ($n=2, n=3, n=4$)?
5. **Q5:** Are conclusions consistent between lengths?
6. **Q6:** Are some polymers structurally unstable or non-convergent under these parameters?

## Physics Configuration

The configuration (`hpc_oracle/configs/surrogate_qc_pilot_qe.yaml`) is strictly an independent computational model designed to estimate the gap. Parameters (PBE functional, 40 Ry cutoff, molecular supercell with 15 Å vacuum) are explicitly declared as `SURROGATE_PROTOCOL_CHOICE`, not `DATASET_MATCHED`.

## Molecular vs Periodic Treatment

Because the generated structures are **capped oligomers**, they must not be simulated as dense continuous solids. They are modeled as **molecular supercells**. This means evaluating the **HOMO-LUMO gap** (Kohn-Sham frontier orbital gap) rather than a true solid-state electronic bandgap. This extracted value is termed the **surrogate electronic gap**.

## The current scientific state is:

> **HANDOFF_PACKAGE_READY** + **CLUSTER_VALIDATION_REQUIRED**

The surrogate pipeline generates strictly controlled input bundles, but no real large-scale calculations have been run locally. The pilot must be transferred to a SLURM cluster, verified, executed, and ingested via the strict result-return protocol. package ready to be copied onto a QE-capable SLURM cluster and executed without changing the scientific protocol.
