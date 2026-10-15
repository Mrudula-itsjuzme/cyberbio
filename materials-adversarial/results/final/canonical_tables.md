# Canonical Evaluation Tables: polyVERSE Bandgap Lineage

Generated from `results/canonical_benchmark_no_leakage.json`.
Target property: Bandgap (eV)

**NOTE**: Drift under randomized SMILES represents representation sensitivity (molecular graph is preserved). MCMC drift represents stress-test drift from true chemistry changes.
Chemistry-changing drift is a measure of robustness to distribution shift, but is NOT proven physical model error without oracle validation.

## Table 1: Random Split

| Model Configuration | Clean RMSE (eV) | Clean MAE (eV) | R^2 | Rand-SMILES Drift (eV) [Repr. Preserving] | MCMC Drift (eV) [Chem. Changing Stress-Test] |
|---|---|---|---|---|---|
| Baseline | 0.6007 | 0.4627 | 0.8170 | 0.5991 | 0.2085 |
| MCMCDefended | 0.6101 | 0.4617 | 0.8112 | 0.5916 | 0.1624 |
| RandSmilesAug | 0.5962 | 0.4622 | 0.8197 | 0.3257 | 0.1609 |
| CombinedDefense | 0.6391 | 0.4953 | 0.7928 | 0.3136 | 0.1097 |

## Table 2: Scaffold Split

| Model Configuration | Clean RMSE (eV) | Clean MAE (eV) | R^2 | Rand-SMILES Drift (eV) [Repr. Preserving] | MCMC Drift (eV) [Chem. Changing Stress-Test] |
|---|---|---|---|---|---|
| Baseline | 0.6998 | 0.5149 | 0.7730 | 0.8968 | 0.2218 |
| MCMCDefended | 0.7184 | 0.5555 | 0.7608 | 0.6710 | 0.1807 |
| RandSmilesAug | 0.6630 | 0.4956 | 0.7963 | 0.4079 | 0.2047 |
| CombinedDefense | 0.6803 | 0.4961 | 0.7855 | 0.3776 | 0.1816 |
