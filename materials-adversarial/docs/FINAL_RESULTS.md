# Final Canonical Results

## Authoritative Results Table

| Model | Parameter Count | Clean Validation MAE | RMSE | R² | Equivalent-SMILES Drift (eV) | Substitution Stress (eV) | Deletion Stress (eV) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ordinary Transformer | ~12M | 0.613 eV | 0.812 | 0.73 | 0.614 | 0.680 | 0.720 |
| Architecture Control | ~12M | 0.382 eV | 0.510 | 0.82 | 0.210 | 0.410 | 0.450 |
| Mixed-Robust Transformer | ~12M | 0.307 eV | 0.420 | 0.88 | 0.150 | 0.320 | 0.360 |
| Augmented Transformer | ~12M | 0.147 eV | 0.201 | 0.94 | 0.080 | 0.190 | 0.210 |
| GraphMPNN | ~5M | 0.411 eV | 0.540 | 0.85 | ≈ 0.000 | 0.420 | 0.460 |

### Adaptive Attack Maximum Bounds

* **Phase 12B Repaired Bounded Maximum**: ≈ **3.19 eV**
* **Invalid Search Bug / Edit Creep (NON-CANONICAL)**: 5.961 eV
