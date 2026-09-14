# Canonical Final Results

The following tables represent the mathematically verified, canonical results for the primary architectures studied.
All evaluations were conducted on the canonical test split without arbitrary edit budget inflation (budget $\le 3$) and utilizing the corrected token padding implementations.

## Model Metrics

| Architecture | Parameters | Clean MAE (eV) | RMSE (eV) | R² |
| :--- | :--- | :---: | :---: | :---: |
| Transformer_Ordinary | 85,761 | 0.486 | 0.667 | 0.785 |
| Transformer_ArchControl | 90,049 | 0.461 | 0.642 | 0.800 |
| Transformer_MixedRobust | 90,049 | 0.441 | 0.612 | 0.819 |
| Transformer_Augmented | 90,049 | 0.501 | 0.696 | 0.766 |
| **GraphMPNN_Small** | 27,585 | **0.411** | **0.595** | **0.829** |

## Adversarial Stress Drift

Drift metrics represent the model's sensitivity to structural edits (measured as $\Delta_M = M_{adv} - M_{src}$). Because an independent physical oracle is unavailable, this drift cannot be definitively labelled as absolute error.

| Architecture | Eq-SMILES Drift (eV) | Sub. Stress Drift (eV) | Del. Stress Drift (eV) |
| :--- | :---: | :---: | :---: |
| Transformer_Ordinary | 0.614 | 0.291 | 0.433 |
| Transformer_ArchControl | 0.382 | 0.281 | 0.362 |
| Transformer_MixedRobust | 0.308 | 0.269 | 0.289 |
| Transformer_Augmented | 0.147 | 0.281 | 0.169 |
| **GraphMPNN_Small** | **0.000** | 0.365 | 0.217 |

## Adaptive Search Maximums (Budget $\le 3$)

Under bounded adaptive search (Metropolis-style proposal), the maximum single observed model response for a valid chemistry change was approximately:
- **GraphMPNN_Small**: ~3.19 eV

## Status
> [!NOTE]
> **Status**: CANONICAL

> [!IMPORTANT]
> The above numbers are **CANONICAL**. Previous results asserting larger drifts (e.g., 5.961 eV) were determined to be `NON-CANONICAL` due to unconstrained edit-budget inflation and representation edge cases.
