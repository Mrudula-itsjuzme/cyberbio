# VALIDITY BOUNDARY REAL DATA AUDIT

**Goal**: Inventory existing real candidate artifacts for the validity boundary study.

### Identified Artifacts
- **File**: `results/phase2_paired_benchmark/clean_attack_records.jsonl`
  - **Method**: Contains Random, MCMC, SMC, CEM, Validity-SMC generated candidates.
  - **Candidate Count**: 13,863
  - **Source IDs**: Present (`sample_id`)
  - **Candidate String**: Present (`adversarial_representation`)
  - **Drift**: Present (`absolute_prediction_drift`)
  - **Validator Outcome**: Present (`validity_status`)
  - **Source-relative Descriptors**: Can be reconstructed from `original_representation` vs `adversarial_representation`.
  - **Target-Model Result**: Real predictions present (`adversarial_prediction`, `original_prediction`).

- **File**: `results/phase2_paired_benchmark/attack_candidates.jsonl`
  - Contains candidate parameters and edited positions.

### Selected Cohort
We extract 200 random samples from `clean_attack_records.jsonl` to test the hypothesis without generating any fake data.
