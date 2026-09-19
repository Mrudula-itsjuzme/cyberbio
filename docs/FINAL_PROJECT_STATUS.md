# FINAL PROJECT STATUS

## Materials branch status
Complete. All regression models, shallow baselines, and attacks verified. Historical defense transfer correctly invalidated due to vocabulary mismatch.

## Bio-Cyber V1 status
Invalidated. Trivialized by 3-mer leakage.

## Bio-Cyber V2 status
Invalidated. Trivialized by composite 5-mer motifs.

## Bio-Cyber V3 status
Complete. V3 substantially reduces the shallow compositional shortcuts identified in V1 and V2. No tested shallow baseline substantially solved V3.

## Model results
Complete. CNN_Distance achieved 54.2% ± 0.8% across five seeds (vs 49.9% shuffled), suggesting a weak learnable positional signal.

## Attack status
Complete. Random, MCMC, and Evolutionary attacks implemented and evaluated. Frozen banks hashed and preserved.

## Defense status
Complete. MCMC adversarial training implemented on train set.

## Adaptive-evaluation status
Complete. Adaptive attacks were evaluated against the defended model at matched edit/query budgets, and flip rates and prediction drift were compared with the clean-model attack results.

## Explainability status
Complete. Counterfactuals computed with bootstrap CIs. Explainability attribution shows no preferential targeting, rigorously evaluated via Wilcoxon test and matched-random background controls.

## Historical materials limitation
Complete. Documented as INVALID_MODEL_RECONSTRUCTION.

## Test results
Complete. Pytest suites enforce budget counts, alphabet adherence, pool constraints, target consistency, and rigorous SHA-256 formatting.

## Reproducibility status
Complete. Manifest contains true execution environment metrics, seeds, and strict binary SHA-256 hashes.

## Remaining optional work
- REQUIRED: None.
- OPTIONAL: larger Transformer tuning, more attack families, larger V3 model study, external/material oracle experiments.
