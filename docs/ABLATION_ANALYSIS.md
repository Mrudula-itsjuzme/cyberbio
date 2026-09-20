# ABLATION ANALYSIS

## Experimental Setup Validity
The `results/final/ablation_results.csv` table represents a valid ablation study because all configurations:
- Use the same architecture (Transformer)
- Use the same data split (RandomSplit)
- Are aggregated over 5 random seeds (matching seeds across configurations)
- Share the same evaluation protocol and attack budget for MCMC edits

## Which training strategy improvides evidence robustness most consistently?
Based on the experimental data, the **CombinedDefense** strategy provides the most consistent improvement across both representation-preserving (Random SMILES) and chemistry-changing (MCMC) perturbations.

## Does defense overfit to the attack used during training?
Yes, partially. MCMC-Defended models show better performance against MCMC attacks but remain highly vulnerable to Random SMILES representation changes. Conversely, RandSmilesAug significantly improvides evidence robustness against Random SMILES but still allows some drift under MCMC.

## Is there cross-attack transfer?
Cross-attack transfer is weak. Defending against MCMC alone does not resolve representation instability. Defending against representation changes (RandSMILES) reduces MCMC drift slightly (because MCMC explores the same token space), but combined defense is performs best.

## What clean-performance penalty occurs?
On the Random split, clean RMSE increases from 0.601 (Baseline) to 0.638 (CombinedDefense), indicating a slight accuracy penalty in distribution.
