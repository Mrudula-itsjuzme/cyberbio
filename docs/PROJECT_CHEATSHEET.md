# PROJECT CHEAT SHEET

**PROBLEM:** Transformer sequence models for polymers are brittle to representation changes and local topological edits.
**DATASET:** ~3500 polymers with Bandgap ($E_g$) in eV.
**MODEL:** Two-Branch Transformer Regressor.

**ATTACK 1 (RandomSMILES):** Evaluates representation invariance by generating chemically identical alternative strings.
**ATTACK 2 (MCMC):** Evaluates local structural sensitivity by proposing plausible, constrained token edits (substitutions/insertions/deletions).

**DEFENSE 1 (RandSmilesAug):** Training with randomized representations.
**DEFENSE 2 (CombinedDefense):** Augmentation + MCMC consistency training.

**BEST VERIFIED NUMBERS:**
- Clean Baseline RMSE (Scaffold): 0.700 eV
- Defended RMSE (Scaffold): 0.635 eV
- RandomSMILES Drift (Baseline): 0.897 eV
- RandomSMILES Drift (Defended): 0.378 eV
- MCMC Drift (Baseline): 0.222 eV
- MCMC Drift (Defended): 0.182 eV

**CORE CONTRIBUTION:** Identifying that representation sensitivity and local structural sensitivity are orthogonal failure modes, and demonstrating that data augmentation acts as an OOD structural regularizer.

**LIMITATION:** Lack of a DFT computational oracle prevents us from guaranteeing that MCMC adversarial structures retain their physical ground-truth bandgap.

**FUTURE WORK:** Integration of quantum chemistry (DFT) validation and graph neural networks (GNNs).

**10 THINGS TO REMEMBER FOR VIVA:**
1. Units are always eV.
2. Property is Bandgap.
3. RDKit provides syntactic validity, not physical reality.
4. Scaffold split measures OOD generalization; Random split measures interpolation.
5. Augmentation fixes representation drift *and* improvides evidence Scaffold RMSE.
6. MCMC defense does not fix RandomSMILES drift.
7. We measure *model consistency*, not absolute physical robustness.
8. We bounded MCMC attacks with Tanimoto similarity and MW limits.
9. Sequence length is a correlational proxy shortcut for the model.
10. Defenses were run across multiple seeds to ensure statistical significance.
