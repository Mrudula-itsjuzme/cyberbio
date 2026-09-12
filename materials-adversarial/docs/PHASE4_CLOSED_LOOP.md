# Phase 4 — Closed-Loop Adversarial Evaluation

## Status

**IMPLEMENTED, NOT YET EXECUTED.** No Phase 4 numerical result is claimed in
this document or in the README. A run necessarily trains three new defended
models and must be started explicitly with a new output directory.

## Research question

Does a defense become robust to the failure mode it was trained against, or
does it only suppress the attack distribution used to train it?

The experiment evaluates one clean model and three defended models trained on
randomization, model-guided stochastic-search (“MCMC”), or a mixed attack set.
Every model is evaluated against randomization, probabilistic MCMC,
substitution, insertion, deletion and rearrangement.

## Scientific taxonomy

| Category | Attacks | Label/evaluation rule |
|---|---|---|
| Representation-preserving control | RDKit SMILES randomization | May inherit the measured target after graph equivalence; threshold attack success is meaningful |
| Chemically valid but target-changing perturbation | substitution, insertion, deletion, rearrangement | Must not inherit Tg/bandgap; training uses an explicitly labeled clean-model teacher prediction; threshold rate is only stress exceedance |
| Adversarial stress test | probabilistic MCMC | Same teacher-prediction rule; adaptive evaluation is unpaired and is not a validated MCMC sampling claim |

The teacher target asks whether a defended model preserves the clean model's
local response under stress. It is **not** a measured physical-property label
and must never be reported as one.

## Pairing policy

- Randomization and the four fixed edit families are generated once and scored
  against every model. Candidate identity is exact and per-example defense
  deltas are paired.
- Probabilistic MCMC is model-guided. It is regenerated against each defended
  model to implement a genuine adaptive re-attack. Those cells are explicitly
  `adaptive_unpaired`; no paired inference or significance test is emitted.
- RDKit randomization now uses RDKit's explicit random seed API, so the primary
  control candidate bank is reproducible from its recorded seed.

## Metrics and uncertainty

Each cell records clean MAE, mean/median/p90/p95/max absolute prediction drift,
validity rate, label-preserving attack success where meaningful, stress
threshold exceedance otherwise, defense delta, robustness transfer and clean
performance tradeoff.

Confidence intervals are 95% percentile bootstrap intervals clustered by source
polymer. Candidates from the same polymer are never treated as independent.
Intervals are omitted when fewer than the configured minimum number of source
polymers is available. No bootstrap interval is reported for maximum drift and
no significance test is manufactured.

## Immutable outputs

Every run requires a path that does not already exist. The implementation
refuses to overwrite it. A successful run writes:

- `attack_candidates/` — fixed banks or per-model adaptive MCMC banks;
- `raw_attack_records/` — complete records for every matrix cell;
- `training_attacks/` and `training_data/` — defense provenance and label basis;
- `clean_predictions/` — per-example clean errors;
- `paired_results/` — fixed-family per-example defense deltas;
- `summary.json` — full metrics and confidence intervals;
- `robustness_transfer_matrix.csv` — requested attack-by-defense matrix of mean drift;
- `robustness_transfer_cells.csv` — long-form matrix with transfer/tradeoff fields;
- `config_snapshot.json` and `reproducibility_metadata.json` — configuration,
  source state, seeds, package versions, input hashes and checkpoint hashes.

## Execution

From `materials-adversarial/`, with the optional model dependencies installed:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_closed_loop.py \
  --run-dir results/phase4/<unique-run-id> \
  --config configs/closed_loop.yaml \
  --clean-model-dir results/models/transformer_regressor
```

The shipped configuration explicitly names the active
`polyverse_bandgap` lineage. Execution stops if it does not match
`configs/dataset.yaml`. Historical Tg artifacts therefore cannot be silently
mixed into this run.
