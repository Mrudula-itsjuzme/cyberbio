# Framework V2 Attack Results

Two runs live in this file. Only the second one may be cited.

## 1. Developmental run — `DEVELOPMENTAL_UNVERIFIED`, DEPRECATED

Verdict of the forensic audit: **MULTIPLE_PROTOCOL_ERRORS**
(`docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md`). These numbers are invalid and must not be
quoted: the predictor was `float(hash(smiles) % 1000) / 100.0`, 25/30 sources came from
**train** and 5/30 from **test**, and 482/540 runs exceeded their declared query budget.

Frozen artifact: `results/framework_v2/developmental_run_1/attack_results.csv`.

| Strategy | Operator | Q | Mean Drift (invalid) | Max Drift (invalid) |
|---|---|---|---|---|
| Random | SimpleSubst | 10 | 6.351 | 9.590 |
| Random | Motif | 10 | 5.422 | 9.520 |
| Greedy | SimpleSubst | 10 | 6.434 | 9.590 |
| Greedy | Motif | 10 | 5.740 | 9.520 |
| Evolutionary | SimpleSubst | 10 | 6.438 | 9.590 |
| Evolutionary | Motif | 10 | 5.740 | 9.520 |
| Random | SimpleSubst | 20 | 6.496 | 9.590 |
| Random | Motif | 20 | 5.674 | 9.520 |
| Greedy | SimpleSubst | 20 | 6.751 | 9.590 |
| Greedy | Motif | 20 | 5.892 | 9.520 |
| Evolutionary | SimpleSubst | 20 | 6.767 | 9.590 |
| Evolutionary | Motif | 20 | 5.936 | 9.520 |
| Random | SimpleSubst | 50 | 6.707 | 9.520 |
| Random | Motif | 50 | 5.742 | 9.520 |
| Greedy | SimpleSubst | 50 | 6.892 | 9.590 |
| Greedy | Motif | 50 | 5.917 | 9.520 |
| Evolutionary | SimpleSubst | 50 | 6.950 | 9.590 |
| Evolutionary | Motif | 50 | 5.938 | 9.520 |

## 2. Verified run 1 — `verified_run_1`

Validation-only 30-source stratified subset, canonical GraphMPNN
(`1676e34bb7669a27…` source manifest), every model call
charged through one budgeted scoring path.

* split membership: train 0,
  validation 30,
  test 0
* splits sha256 `4a9d28252d058c0c2e39cf70eb10dd2f79ad57df4e842c3f917cdfe9830ccd02`
* stratification {'low': 10, 'medium': 10, 'high': 10}, unique Murcko scaffolds
  6, unique scaffold groups
  21, unique atom counts 20

### Clean model verification (preflight)

| metric | measured | canonical | within tolerance |
|---|---|---|---|
| MAE (eV) | 0.411190 | 0.411190 | yes |
| RMSE (eV) | 0.595369 | 0.595369 | yes |
| R² | 0.828639 | 0.828639 | yes |
| max abs diff vs Phase 11C evaluator (20 samples) | 0.0e+00 | 0 | yes |

### Query accounting

Every run satisfies `predictor_calls == 1 + queries_used` and `queries_used <= Q`
(source scored once, uncharged). Violations: **0**
across 870 runs.

### Untargeted search strategies (operator: SimpleSubstitution, 3-edit budget)

| Strategy | Q | Runs | Mean drift (eV) | Median (eV) | p95 (eV) | Max (eV) | Mean queries | Unique cand. rate | Max atom edits |
|---|---|---|---|---|---|---|---|---|---|
| evolutionary | 10 | 30 | 1.207 | 1.086 | 2.299 | 2.422 | 10.000 | 1.000 | 2 |
| evolutionary | 20 | 30 | 1.545 | 1.511 | 2.556 | 2.942 | 18.333 | 1.000 | 3 |
| evolutionary | 50 | 30 | 1.583 | 1.548 | 2.556 | 2.942 | 19.767 | 1.000 | 3 |
| greedy | 10 | 30 | 0.947 | 0.948 | 1.667 | 1.894 | 10.000 | 1.000 | 1 |
| greedy | 20 | 30 | 1.188 | 1.172 | 2.242 | 2.495 | 20.000 | 1.000 | 3 |
| greedy | 50 | 30 | 1.703 | 1.656 | 2.833 | 3.055 | 47.467 | 1.000 | 3 |
| metropolis | 10 | 30 | 1.054 | 1.005 | 1.809 | 1.897 | 10.000 | 1.000 | 3 |
| metropolis | 20 | 30 | 1.278 | 1.225 | 2.186 | 2.652 | 20.000 | 1.000 | 3 |
| metropolis | 50 | 30 | 1.552 | 1.577 | 2.811 | 3.055 | 49.467 | 1.000 | 3 |
| random | 10 | 30 | 0.967 | 0.931 | 1.807 | 1.894 | 10.000 | 1.000 | 1 |
| random | 20 | 30 | 1.090 | 1.012 | 1.807 | 1.894 | 18.833 | 1.000 | 1 |
| random | 50 | 30 | 1.166 | 1.285 | 1.807 | 1.894 | 30.567 | 1.000 | 1 |

### Targeted objectives (Q=50) — DIRECTIONAL_SUCCESS_RATE

`DIRECTIONAL_SUCCESS_RATE` is the fraction of sources whose prediction moved in the
requested direction. It is a sign test, not attack success.

| Objective | Strategy | Directional success rate | Mean signed change (eV) | Median signed change (eV) | p10 (eV) | p90 (eV) | Mean queries |
|---|---|---|---|---|---|---|---|
| TARGET_DECREASE | evolutionary | 1.000 | -1.481 | -1.438 | -2.298 | -0.873 | 20.100 |
| TARGET_DECREASE | greedy | 1.000 | -1.571 | -1.525 | -2.341 | -0.586 | 47.533 |
| TARGET_DECREASE | metropolis | 1.000 | -1.464 | -1.521 | -2.120 | -0.666 | 48.200 |
| TARGET_DECREASE | random | 1.000 | -0.980 | -0.766 | -1.632 | -0.424 | 30.567 |
| TARGET_INCREASE | evolutionary | 1.000 | 0.920 | 0.737 | 0.257 | 1.687 | 28.300 |
| TARGET_INCREASE | greedy | 1.000 | 0.926 | 0.751 | 0.295 | 1.652 | 47.467 |
| TARGET_INCREASE | metropolis | 1.000 | 0.866 | 0.785 | 0.230 | 1.628 | 47.033 |
| TARGET_INCREASE | random | 1.000 | 0.625 | 0.438 | 0.226 | 1.364 | 30.567 |

### One-edit sanity gate

| Sources | Candidates | Mean (eV) | Median (eV) | p95 (eV) | Max (eV) |
|---|---|---|---|---|---|
| 10 | 363 | 0.206 | 0.097 | 0.689 | 1.894 |

Canonical bounded maximum reference: 3.19 eV.
Forensic reference regime: mean 0.213, median 0.104, max 1.547 eV.

### Cross-model transfer (one-way: GraphMPNN-chosen candidates re-scored by the Transformer)

| quantity | value |
|---|---|
| candidates | 30 |
| mean abs GraphMPNN Δ (eV) | 1.703 |
| mean abs Transformer Δ (eV) | 1.370 |
| signed agreement rate | 0.867 |
| Pearson r | 0.699 |
| Spearman r | 0.573 |
| transfer ratio (Transformer / GraphMPNN) | 0.805 |

### Phase 12B bridge (Q ∈ {10, 50}, all 30 verified sources)

| Method | Phase 12B mean (eV) | V2 mean (eV) | Phase 12B max (eV) | V2 max (eV) |
|---|---|---|---|---|
| greedy | 1.453 | 1.325 | 4.499 | 3.055 |
| metropolis | 1.084 | 1.303 | 2.508 | 3.055 |
| random | 1.279 | 1.067 | 2.992 | 1.894 |

Residual differences are expected and explained by operator semantics, proposal space,
search implementation and RNG order (see `phase12b_bridge_summary.json`). Direct
numerical equivalence is NOT claimed.

**Metric naming:** chemistry-changing numbers here are `PREDICTION DRIFT`, never
"adversarial error".
