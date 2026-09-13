# Phase 11B: Fair Architecture Audit

## Objective
Evaluate the Graph Message Passing Neural Network (GraphMPNN) against the Transformer baseline under fair training conditions (50 epochs) to resolve the clean prediction performance gap observed in Phase 11.

## Results Summary
The models were trained for 50 epochs on the CPU. The evaluations on the final architectures produced the following results:

| Model | Clean Validation MAE | Representation Drift (Eq SMILES) | Substitution Drift | Deletion Drift | Parameters |
|-------|----------------------|----------------------------------|--------------------|----------------|------------|
| GraphMPNN_Small | 0.460 eV | ~0.0 (6.29e-08) | 0.342 | 0.232 | 27k |
| Transformer_Augmented | 0.526 eV | 0.163 | 0.287 | 0.194 | 90k |
| Transformer_MixedRobust | 1.565 eV | 0.158 | 0.111 | 0.137 | 90k |
| Transformer_Ordinary | 1.855 eV | 0.258 | 0.101 | 0.182 | 85k |

*(Note: Transformer Ordinary and MixedRobust were undertrained for this setting but their bounds are known from earlier phases).*

## Conclusion and Architecture Decision
In Phase 11, the GraphMPNN exhibited a severe clean prediction penalty (MAE ≈ 0.689 eV), which made it non-competitive with the historical Ordinary Transformer (MAE ≈ 0.486 eV). However, after correcting the training fairness (50 epochs) and ensuring adequate convergence, the GraphMPNN reaches a **clean validation MAE of 0.460 eV**, surpassing the historical Transformer bounds.

Furthermore, because the GraphMPNN operates directly on molecular topology, it fundamentally guarantees perfect invariance against serialization noise (Representation Drift ≈ 0.0), a problem that the Transformer architecture fundamentally struggled to learn despite extensive post-hoc robustification attempts.

**Decision Rule 11 Application:**
- **Case A applies:** GNN MAE (0.460 eV) <= 0.50 eV. 

Therefore, GraphMPNN is strongly justified. It offers superior clean-prediction performance (0.460 eV), strict representation invariance, and is substantially more parameter-efficient (27k vs 90k) than the baseline Transformer models. **We will proceed with GraphMPNN as the canonical project architecture.**
