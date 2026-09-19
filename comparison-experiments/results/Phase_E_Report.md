TEST COUNT
21 automated tests spanning adapters, schemas, parsing, statistics, budgets, edge-cases, and sweep logic.

TEST RESULTS
All 21 tests PASSED. Validation and query accounting is structurally sound and strictly identical across methodologies. Duplicate tracking and diversity metrics correctly function.

SWEEP RUNS COMPLETED
45 runs complete (45 combinations covering random, mcmc, evolutionary across budgets 10,20,50 over 5 seeds for 100 sources each). Total of 4500 sequential attack outcomes generated.

FAILED RUNS
0 

RAW RESULT PATH
`/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/raw/budget_sweep/`

SUMMARY CSV PATH
`/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/summaries/budget_sweep_results.csv`

SUMMARY JSON PATH
`/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/summaries/budget_sweep_results.json`

PLOTS CREATED
Generated 7 plots in `/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/plots/budget_sweep/`:
- `mean_drift_vs_budget.png`
- `median_drift_vs_budget.png`
- `success_rate_vs_budget.png`
- `validity_rate_vs_budget.png`
- `runtime_vs_budget.png`
- `queries_vs_budget.png`
- `similarity_vs_drift.png`

EVOLUTIONARY CONFIG SELECTED
Tuning completed against internal validation set (excluding frozen_source_ids). 
Parameters serialized to `results/evolutionary_tuning/evolutionary_config.json`.
Selected Config: `{"population_size": 5, "mutation_rate": 0.2, "elitism_count": 1}`

STATISTICAL METHODS USED
- Standard metrics (mean, median, p90, std-dev)
- 95% Confidence Intervals (Bootstrap with n=1000 resamples)
- Wilcoxon Signed-Rank Test (Paired per-source predictions)
Outputted to `/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/summaries/sweep_statistical_analysis.txt`.

KNOWN LIMITATIONS
- The evolutionary models at tight budgets (e.g. 10 queries) tend to underperform compared to brute MCMC because population generation costs immediately eat up constraints.
- `validity_rate` metric tends to stay artificially near `1.0` due to `is_valid_plausible` brute-force masking before evaluating candidates against the `model.predict()`, inflating perceived structural efficiency.
