# Surrogate QC Cluster Handoff Guide

This guide details the procedure for moving the surrogate QC pilot to an HPC cluster, executing it, and returning the results to the local environment.

## 1. Transfer the Bundle

The bundle `oracle_surrogate_qc_bundle` is self-contained. Copy it to your SLURM cluster:
```bash
scp -r oracle_surrogate_qc_bundle user@cluster.address:~/
```

## 2. Verify Cluster Environment (Pre-flight Check)

Before submitting any jobs, verify the environment and pseudopotentials.
```bash
cd oracle_surrogate_qc_bundle
python scripts/verify_cluster_environment.py
```
This script will:
- Check for `pw.x`.
- Scan `qe_inputs/pseudo` for required pseudopotentials.
- Verify cutoffs.
- Generate `cluster_environment.json` and `cluster_pseudopotential_manifest.json`.

If pseudopotentials are missing, download them (SSSP efficiency 1.2) into `qe_inputs/pseudo/`.
If cutoffs mismatch the 40 Ry lock, the script will block execution.

## 3. Execute SLURM Array

Once verification is successful (`EXECUTION_READY`), submit the array:
```bash
sbatch slurm/submit_surrogate_qc_array.sh
```
This will run the 9 jobs. Outputs go to `out/<JOB_ID>/`.

## 4. Package Results

After all jobs complete (successful or not):
```bash
python scripts/package_surrogate_qc_results.py
```
This creates `surrogate_qc_results_bundle` containing all outputs, logs, environment snapshots, and cryptographic hashes.

## 5. Return Results

Copy the results bundle back to your local machine:
```bash
scp -r user@cluster.address:~/oracle_surrogate_qc_bundle/surrogate_qc_results_bundle /path/to/local/oracle_surrogate_qc_bundle/
```

Proceed to the ingestion steps locally.
