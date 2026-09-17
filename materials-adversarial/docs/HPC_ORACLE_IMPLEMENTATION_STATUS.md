# HPC Oracle Implementation Status

## Components

| Component | Status | Notes |
| --- | --- | --- |
| `hpc_oracle.backend.base` | IMPLEMENTED | Structured typed contract established |
| `hpc_oracle.backend.qe` | PARTIAL | Executable check and skeleton writing implemented. Blocked on real QC physics parameters. |
| `hpc_oracle.backend.vasp` | PARTIAL | Executable check and skeleton writing implemented. Blocked on licensed POTCAR. |
| `hpc_oracle.structure.prep` | IMPLEMENTED | Construction hashing and provenance logic is separate from QC backend. |
| `hpc_oracle.manifests.calibration` | IMPLEMENTED | Supports dataset mapping to deterministic hash job IDs. |
| `hpc_oracle.manifests.adversarial` | IMPLEMENTED | Generates linked source/candidate pairs for error calculations. |
| `hpc_oracle.slurm.generator` | IMPLEMENTED | Highly parameterized script builder supporting arrays. |
| `scripts.check_oracle_backend` | IMPLEMENTED | Verifies QE/VASP setup locally. |
| `scripts.export_oracle_jobs` | IMPLEMENTED | Dry-run creation of job submission bundles. |
| `scripts.ingest_oracle_results` | IMPLEMENTED | Converts output into structured JSON for analysis. |
| `scripts.analyze_oracle_calibration`| IMPLEMENTED | Extracts R², RMSE, MAE, Pearson metrics. |
| `scripts.analyze_oracle_adversarial`| IMPLEMENTED | Extracts actual $\Delta_T, E_G, E_S$ terms. |
| `hpc_oracle.analysis.uncertainty` | IMPLEMENTED | Explicit structure for tracking `UNKNOWN` variances. |

## External Blockers
- Real `pseudopotential` (QE) or `POTCAR` (VASP) required.
- Actual physical parameter tuning (Functional, KPOINTS, Cutoff, etc.).
- Active SLURM cluster to submit jobs to.
