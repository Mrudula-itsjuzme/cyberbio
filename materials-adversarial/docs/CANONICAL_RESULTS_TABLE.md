# Canonical Results

This table represents the authoritative, mathematically verified results from the final canonical evaluation (Phase 11C and Phase 12B). All summary documents must reference this table.

| Model | Checkpoint SHA256 | Parameters | Validation MAE (eV) | RMSE (eV) | R² | Equivalent-SMILES Drift (eV) | Substitution Stress Drift (eV) | Deletion Stress Drift (eV) | Canonical Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Transformer_Ordinary | `0b1bb3f62ca72a89...` | 85,761 | 0.486 | 0.667 | 0.785 | 0.614 | 0.291 | 0.433 | CANONICAL_BASELINE |
| Transformer_ArchControl | `fef6c0e9d7883d90...` | 90,049 | 0.461 | 0.642 | 0.800 | 0.382 | 0.281 | 0.362 | HISTORICAL |
| Transformer_MixedRobust | `b81aa245444431a3...` | 90,049 | 0.441 | 0.612 | 0.819 | 0.308 | 0.269 | 0.289 | HISTORICAL |
| Transformer_Augmented | `d06da5debd6e2a12...` | 90,049 | 0.501 | 0.696 | 0.766 | 0.147 | 0.281 | 0.169 | HISTORICAL |
| **GraphMPNN_Small** | `3fd339c5ebd0d700...` | 27,585 | **0.411** | **0.595** | **0.829** | **0.000** | **0.365** | **0.217** | **CANONICAL_PREDICTOR** |

*Note: GraphMPNN equivalent-SMILES drift is strictly zero by structural definition (computed as 4.8e-08 due to float precision). The maximum bounded chemistry-changing drift observed for GraphMPNN via adaptive search is ~3.19 eV.*
