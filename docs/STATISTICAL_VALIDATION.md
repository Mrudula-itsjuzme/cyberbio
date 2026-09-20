# STATISTICAL VALIDATION

## Rationale
To ensure the defense improvements are not statistical noise or seed-specific artifacts, we evaluated the results across $N=3$ independent training initializations (seeds 42, 123, 456). Given the small sample size of initializations, we focus on descriptive robustness statistics (mean, std) and large effect sizes rather than calculating potentially misleading $p$-values or artificially narrow confidence intervals.

## Effect Magnitude
The reductions in drift due to defense interventions are large in magnitude compared to the standard deviation across seeds:
- The reduction in Random-SMILES drift by `RandSmilesAug` is far greater than the inter-seed variance (effect magnitude $>0.4$ eV against $\sigma \approx 0.05$ eV).
- The reduction in MCMC drift by `CombinedDefense` relative to the baseline similarly dwarfs inter-seed noise.

These observations confirm that the architectural/defense interventions systematically alter the model's robustness profile.

The exact aggregations are stored in `results/final/ablation_results.csv`.
