# Project Status

This document defines the exact frozen state of the Phase 1 + Phase 2 milestone of the Materials Adversarial Learning framework. No further experiments are currently running.

## COMPLETED
- **Dataset Preparation**: Successfully migrated from OpenPoly to polyVERSE Bandgap, extracting 4,209 usable records.
- **Baseline Model**: Configured and trained a Transformer regression model (Clean Test MAE: `0.4619 eV`).
- **Attack Engine**: Developed token-level perturbation algorithms (Substitution, Rearrangement, Insertion, Deletion, Randomization, MCMC).
- **Validity Filtering**: Implemented strict RDKit parsing; sequences violating fundamental valency were disqualified (69.4% validity rate).
- **Attack Evaluation (Phase 1)**: Demonstrated systemic baseline vulnerability, with 4,232 successful validation attacks exhibiting a mean absolute drift of ~0.48 eV and a maximum prediction drift of 5.47 eV.
- **Adversarial Training (Phase 2)**: Filtered 79,825 candidate attacks into a strictly label-preserving set of 11,300 substitution and rearrangement variants, successfully training the `transformer_defended` model.
- **Defended-Model Evaluation**: Evaluated Phase 2 model robustness using symmetrical candidate pools on the sealed Test Split.
- **Research Documentation**: Authored comprehensive scientific reports detailing architectural assumptions, label-preservation policies, and evaluation bounds.

## CURRENT FINDING
- **Attack-Specific Robustness Demonstrated**: The defense significantly immunized the model against the attack families seen during training (Substitution success dropped from 20.13% to 7.53%).
- **Clean Performance Preserved**: The defense did not degrade standard predictive capacity (Clean Test MAE marginally improved from `0.4619 eV` to `0.4601 eV`).
- **Unseen-Attack Generalization Not Demonstrated**: The defense completely failed to transfer robustness to unseen attack families (Insertion and Deletion success rates remained stagnant at ~32%).

## NOT YET DONE
- **Probabilistic/MCMC Experiments**: MCMC attacks were generated but excluded from training augmentation due to safety concerns regarding label-preservation.
- **DFT/Physics Validation**: Currently relying on strict heuristic constraints for label preservation. No *ab initio* validation exists to confirm the true Bandgap of the generated adversarial strings.
- **Experimental Material Validation**: The vulnerabilities are mathematically proven in the model's representation space, but we lack laboratory synthesis data to confirm whether these representations map to synthesizable physical materials with the predicted properties.
- **Additional Property Datasets**: Architecture is generic, but large-scale tuning on Tg or Density has not been conducted.
- **Broader Generalized-Defense Experiments**: Designing structural augmentation policies that effectively generalize against sequence length alterations (insertions/deletions).
