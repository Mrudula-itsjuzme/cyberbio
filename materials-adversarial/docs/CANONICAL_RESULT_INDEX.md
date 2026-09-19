# CANONICAL RESULT INDEX

| Claim | Artifact | Script | Config | Dataset Split | Seed(s) | Metric |
|---|---|---|---|---|---|---|
| Random-SMILES baseline drift | results/canonical_benchmark_no_leakage.json | scripts/run_comprehensive_benchmark_suite.py | configs/canonical_benchmark.json | scaffold | 42, 123, 2026, 777, 999 | mean drift, RMSE |
| MCMC attack transferability | results/framework_v2/mcmc_transfer.json | scripts/run_canonical_reproduction.py | configs/mcmc_attack.json | scaffold | 42, 123, 2026, 777, 999 | success rate |
| Defense Effectiveness | results/before_after_defense_experiment_results.json | scripts/run_before_after_defense_experiment.py | configs/defense_train.json | scaffold | 42, 123, 2026, 777, 999 | paired robustness delta |

