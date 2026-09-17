# HPC Oracle Execution Guide

This guide details the procedure for executing the `oracle_surrogate_qc_bundle` on a Quantum ESPRESSO capable SLURM cluster.

## Pre-requisites

- A SLURM-managed HPC cluster.
- Quantum ESPRESSO (`pw.x`) available in the environment.
- Required SSSP efficiency 1.2 pseudopotentials matching the elements in `manifests/required_pseudopotentials.csv` placed in `qe_inputs/pseudo/`.

## 1. Validate Bundle

Before submission, ensure the bundle is intact and configured properly.

```bash
python scripts/validate_surrogate_qc_bundle.py
```
This script will verify:
- Hashes of all files match `HASHES.json`.
- Exactly 9 jobs are present in the manifest.
- All structures and configurations are valid.

## 2. Submit SLURM Array

A standard SLURM submission script should be adapted to iterate over the input files in `qe_inputs/`. 

Example SLURM execution loop:
```bash
#!/bin/bash
#SBATCH --job-name=surrogate_qc
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=32

module load quantum-espresso

MANIFEST="manifests/surrogate_qc_pilot_9jobs.csv"
# Extract job ID based on SLURM_ARRAY_TASK_ID
# ...
pw.x -i qe_inputs/${JOB_ID}.in > out/${JOB_ID}.out
```

## 3. Monitor and Collect Outputs

Wait for all array tasks to finish. Ensure the `.out` files are collected.

## 4. Retrieve Results

Copy the `out/` directory and any relevant `.csv` manifests back to the local repository. 

## 5. Ingest and Analyze

Run the analysis pipeline locally on the retrieved outputs.

```bash
python scripts/analyze_surrogate_qc_pilot.py --outputs path/to/retrieved/out/
```

## Failure Recovery

- **Single Job Failure**: Inspect the `.out` file. Do not silently rerun with changed physics. If a physics change is required, a NEW protocol version must be created locally, generating a new bundle and hashes.
- **SCF Nonconvergence**: Noted by the parser and marked as a failure. 
- **Pseudopotential Mismatch**: Ensure the `pseudo_dir` is correctly populated.
