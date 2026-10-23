# Reusable Interface Audit

Status: experimental audit, 2026-09-20. No canonical files were modified.

## Reusable sources

| Concern | Existing source | Reuse decision |
|---|---|---|
| Dataset/source IDs | `research/comparison-experiments/src/adapters/canonical_dataset.py` and frozen source IDs | Wrap after split/index provenance is asserted; do not copy data. |
| Model | canonical checkpoints plus predictor interfaces | Wrap with a counted prediction gateway; never call the model outside it. |
| Validation | `materials_adv.validation.pipeline.validate` | Reuse directly. Preserve `True`/`False`/`None`; never coerce the result to `bool`. |
| Search contracts | `materials_adv.framework.interfaces` | Reuse concepts (`Candidate`, operator edit cost, injected identity), while keeping lab result/accounting types isolated. |
| Random/evolutionary/MCMC | `comparison-experiments` adapters | Repair behind lab adapters; do not rewrite mature algorithms. |
| Metrics | canonical attack metrics and comparison statistics | Reuse only after population and missing-value semantics are checked. |
| Attribution | representation-attribution experiment | Attribution cost must be charged separately and controls retained. |

## Blocking findings

1. `CanonicalValidatorAdapter.is_valid_plausible` calls `bool(result)`. The current canonical `ValidationResult.__bool__` deliberately raises `TypeError`. This adapter cannot be reused unchanged.
2. `CanonicalMCMCAdapter` makes a final `model.predict(best_candidate)` without charging the query ledger. This violates matched target-model budgets.
3. The MCMC proxy truncates an over-budget batch, which can change return cardinality and algorithm behavior. Budget enforcement must happen before invocation.
4. The comparison schema collapses RDKit and constraints to booleans; the lab must retain unchecked values as missing.
5. The random baseline proposes from the current best while its metadata does not report all proposal/validity counters. It needs an accounting adapter before fair use.

## Boundary

This lab lives under `experiments/probabilistic-search-lab`. Canonical datasets,
checkpoints, validators, metrics, and reference artifacts are read-only inputs.
Experimental output must be written under this lab's `results/` directory.
