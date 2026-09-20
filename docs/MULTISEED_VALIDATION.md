# MULTI-SEED VALIDATION

## Methodology
The final defense configurations (Baseline, RandSmilesAug, MCMCDefended, CombinedDefense) were run across three independent deterministic seeds (`42`, `123`, `456`) on identical Scaffold and Random splits.

## Findings
The improvements observed in the preliminary single-seed baseline remain highly stable across initializations.

1. **Stability:** The variance between seeds for prediction drift is consistently small (std < 0.05 eV for all major metrics).
2. **RandSmilesAug:** Consistently reduces Random-SMILES drift by >50% across all seeds while maintaining or slightly improving clean scaffold RMSE.
3. **MCMCDefended:** Consistently reduces MCMC drift, demonstrating that the adversarial regularization is not a seed-specific artifact.
4. **CombinedDefense:** Consistently achieves the lowest combined prediction drift for both representation-preserving and chemistry-changing attacks across all initializations.

The full numerical breakdown is available in `results/final/multiseed_defense_results.csv` and aggregated as mean $\pm$ std in `results/final/ablation_results.csv`.
