# Surrogate QC Result Return Protocol

This protocol defines the exact procedure for securely ingesting and validating surrogate QC calibration results returned from a remote HPC environment.

## Scientific Contract

Results are only considered valid if:
1. Cryptographic hashes of all output files match the returned `RESULT_HASHES.json`.
2. The `SURROGATE_PROTOCOL_LOCK.json` returned by the cluster exactly matches the local physics lock.
3. The cluster environment snapshot shows no unauthorized runtime substitutions.
4. The pseudopotential manifest shows that required cutoffs align with the generated inputs.

## 1. Validation

Once the `surrogate_qc_results_bundle` is placed inside `oracle_surrogate_qc_bundle/` locally, run the validation check:

```bash
python scripts/validate_surrogate_qc_results.py
```

This script will block if any hashes mismatch, if the protocol lock was altered, or if inputs are missing.

## 2. Ingestion

After validation succeeds, run the ingestion script:

```bash
python scripts/ingest_surrogate_qc_results.py
```

This uses the canonical `SurrogateParser` to extract the `band_gap_ev` and `total_energy_ryd` from every `qe.out` file. It outputs parsed records to `results/surrogate_qc_ingested/parsed_gaps.csv`.

## 3. Length Convergence Analysis

Once results are ingested, length convergence trends (n=2, n=3, n=4) can be evaluated to determine whether the surrogate polymer chain geometries produce stable, predictable asymptotic bandgaps. This step will dictate if the surrogate can proceed to adversarial pairs.
