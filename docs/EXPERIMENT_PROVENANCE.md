# EXPERIMENT PROVENANCE

## A. Prediction Task
Polymer property regression from sequence representations.

## B. Target Variable
Bandgap ($E_g$)

## C. Target Units
Electron volts (eV)

## D. Dataset Columns
- `original_representation`: Tokenized PSMILES string.
- `property_value`: Target bandgap in eV.

## E. Train/Test Split Logic
Random splitting via `scikit-learn` `train_test_split`, ensuring an unconditional IID distribution.

## F. Scaffold Split Logic
Groups molecules by their canonical RDKit SMILES string (`scaffold_key`). Ensures that chemically equivalent serializations are not split across training and testing sets, enforcing out-of-distribution (OOD) structural generalization.

## G. Model Architecture
Two-Branch Transformer Regressor (`TwoBranchTransformerRegressorModel`). Uses an embedding layer, positional encoding, and a Transformer encoder followed by a linear regression head.

## H. Tokenizer/Vocabulary Handling
Character-level SMILES tokenization combined with semantic token matching for brackets `[...]` and double-letter elements (e.g., `Cl`, `Br`).

## I. Defense Configurations
1. **Baseline**: Standard MSE training on original dataset.
2. **RandSmilesAug**: Training with randomized SMILES augmentation.
3. **MCMCDefended**: Adversarial consistency training using MCMC perturbed variants.
4. **CombinedDefense**: Employs both randomized SMILES augmentation and MCMC consistency regularization.

## J. Attack Budgets
Typically 1-3 token modifications for MCMC attacks to ensure localized edits.

## K. Evaluation Metrics
- Clean Performance: MAE, RMSE, $R^2$.
- Robustness: Mean Prediction Drift (eV).

## L. Seed Configuration
Historically single-seed (42). Moving to a multi-seed robust protocol (42, 123, 456).

## M. Leakage Controls
Scaffold splitting explicitly groups identical molecules prior to splitting to prevent data leakage between train and test sets.
