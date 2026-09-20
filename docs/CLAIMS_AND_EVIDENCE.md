# FINAL CLAIMS AND EVIDENCE MATRIX

### Claim 1: The Transformer is sensitive to alternative chemically equivalent serializations.
- **Evidence**: Mean prediction drift of 0.897 eV on Scaffold Split under RandomSMILES attack (Baseline model).
- **Source**: `results/final/multiseed_defense_results.csv`
- **Seed Count**: 3 (seeds 42, 123, 456).
- **Interpretation**: The model acts on sequence tokens, failing to completely internalize the graph isomorphism.
- **Limitations**: Only tested on PSMILES syntax.

### Claim 2: Random-SMILES augmentation reduces representation sensitivity.
- **Evidence**: RandSmilesAug configuration reduces RandomSMILES drift to ~0.408 eV.
- **Source**: `results/final/ablation_results.csv`
- **Seed Count**: 3
- **Interpretation**: Exposing the model to variable string representations during training acts as a structural regularizer.

### Claim 3: Representation augmentation may also improve scaffold-split predictive generalization.
- **Evidence**: Clean Scaffold RMSE improvides evidence from ~0.700 eV (Baseline) to ~0.663 eV (RandSmilesAug).
- **Interpretation**: Memorizing syntax harms structural generalization; breaking syntax reliance improvides evidence it.

### Claim 4: MCMC perturbations reveal sensitivity to constrained local chemical modifications.
- **Evidence**: MCMC attacks generate ~0.222 eV drift on average, even though changes are restricted to 1-3 tokens and bounded by Tanimoto similarity.
- **Interpretation**: The network is brittle to localized topology edits.
- **What it does NOT establish**: It does not establish that the bandgap of the new molecule is strictly unchanged (no DFT).

### Claim 5: MCMC-oriented defense reduces that measured sensitivity.
- **Evidence**: MCMC drift drops to ~0.160 eV under MCMCDefended training.
- **Interpretation**: Adversarial training successfully forces the model to smooth out its local decision boundary.

### Claim 6: Single-defense strategies transfer imperfectly across attack classes.
- **Evidence**: MCMC defense does not substantially reduce RandomSMILES drift, and RandSmilesAug provides only partial cross-transfer to MCMC robustness.
- **Interpretation**: The failure modes are largely orthogonal.

### Claim 7: CombinedDefense performs best across the evaluated robustness metrics/configurations.
- **Evidence**: Lowest combined drift for both RandomSMILES (~0.378 eV) and MCMC (~0.182 eV).
- **Limitations**: The model is still not completely robust, leaving a ~0.37 eV residual drift.

### Claim 8: Sequence length is associated with model predictions/errors/drift in ways consistent with partial shortcut reliance.
- **Evidence**: Absolute prediction drift correlates with absolute length change during attacks.
- **Interpretation**: The network is highly sensitive to token insertion/deletion, likely using length as a proxy for structural complexity.
