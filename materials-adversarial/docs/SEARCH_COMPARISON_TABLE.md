# Search Strategy Comparison

Generated from `results/framework_v2/verified_run_1/strategy_summary.csv`.
Operator fixed to SimpleSubstitution, edit budget 3, 30 validation-only sources,
canonical GraphMPNN, source prediction uncharged.

| Strategy | Q | Mean drift (eV) | Median (eV) | p95 (eV) | Max (eV) | Mean queries used | Drift per query (eV) | Unique candidate rate | Duplicates | Runs with no accepted edit |
|---|---|---|---|---|---|---|---|---|---|---|
| evolutionary | 10 | 1.207 | 1.086 | 2.299 | 2.422 | 10.000 | 0.121 | 1.000 | 517 | 0 |
| evolutionary | 20 | 1.545 | 1.511 | 2.556 | 2.942 | 18.333 | 0.084 | 1.000 | 1446 | 0 |
| evolutionary | 50 | 1.583 | 1.548 | 2.556 | 2.942 | 19.767 | 0.081 | 1.000 | 1912 | 0 |
| greedy | 10 | 0.947 | 0.948 | 1.667 | 1.894 | 10.000 | 0.095 | 1.000 | 0 | 0 |
| greedy | 20 | 1.188 | 1.172 | 2.242 | 2.495 | 20.000 | 0.059 | 1.000 | 14 | 0 |
| greedy | 50 | 1.703 | 1.656 | 2.833 | 3.055 | 47.467 | 0.038 | 1.000 | 67 | 0 |
| metropolis | 10 | 1.054 | 1.005 | 1.809 | 1.897 | 10.000 | 0.105 | 1.000 | 309 | 0 |
| metropolis | 20 | 1.278 | 1.225 | 2.186 | 2.652 | 20.000 | 0.064 | 1.000 | 1084 | 0 |
| metropolis | 50 | 1.552 | 1.577 | 2.811 | 3.055 | 49.467 | 0.032 | 1.000 | 4962 | 0 |
| random | 10 | 0.967 | 0.931 | 1.807 | 1.894 | 10.000 | 0.097 | 1.000 | 1350 | 0 |
| random | 20 | 1.090 | 1.012 | 1.807 | 1.894 | 18.833 | 0.062 | 1.000 | 5226 | 0 |
| random | 50 | 1.166 | 1.285 | 1.807 | 1.894 | 30.567 | 0.050 | 1.000 | 16247 | 0 |

## Reading notes

* All strategies spend at most `Q` candidate queries; the source is scored once outside
  the budget. Duplicates are dropped BEFORE a query is spent.
* `Runs with no accepted edit` counts sources where no valid candidate was found rather
  than a stronger one.
* Random samples the source neighbourhood without exploitation; Metropolis is a
  temperature-0.1 eV walk, not formal Metropolis-Hastings.
