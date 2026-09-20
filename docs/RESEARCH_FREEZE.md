# RESEARCH FREEZE

**Date:** 2026-09-20
**Status:** CORE RESEARCH FROZEN

## Verified Experiment List
- Baseline Model Evaluation (Random & Scaffold Split)
- RandomSMILES Robustness Attack
- Probabilistic MCMC Constraint-Aware Attack
- RandSmilesAug Defense
- MCMCDefended Adversarial Training
- CombinedDefense

## Verified Seed Set
`[42, 123, 456]`

## Final Tables and Figures
Available as `.csv` programmatic outputs in `results/final/`.

## Known Limitations
The most central limitation of the study is the lack of a Density Functional Theory (DFT) or other physical oracle. Because of this, the MCMC structural perturbations test the network's topological sensitivity, but we cannot legally claim they represent perfectly physically invariant bandgap edits.

## Optional Future Research
No new attack families (LLMs, GANs, RL) should be added for this course project. Future extensions in an active research track would include:
1. True DFT-validated loss functions.
2. Direct comparison to SE(3)-equivariant Graph Neural Networks (GNNs).

**FREEZE DIRECTIVE**: The project is strictly finalized for grading/viva and requires no new algorithmic code.
