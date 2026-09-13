# Phase 2: Specialized Two-Branch Transformer

## 1. Architecture Overview
The Phase 2 model introduces a compact two-branch specialized transformer for polymer Bandgap prediction. It shares a common Transformer encoder and then branches into two explicitly learned specialization heads before a final regression fusion layer.

**Diagram:**
```mermaid
graph TD
    A[PSMILES] --> B[Tokenizer]
    B --> C[Embedding]
    C --> D[Transformer Encoder (Shared)]
    D --> E[Masked Mean Pooling (Shared)]
    
    E --> F[Branch A: Representation Invariance]
    E --> G[Branch B: Chemistry Sensitivity]
    
    F --> H[Concat]
    G --> H
    
    H --> I[Regression Head]
    I --> J[Bandgap (eV)]
```

### Components
- **Shared Encoder:** 64-dim, 2 layers, 4 heads, 128 feedforward dim, dropout 0.1, masked mean pooling.
- **Branch A (f_repr):** `Linear(64 -> 32) -> ReLU -> LayerNorm`
- **Branch B (f_chem):** `Linear(64 -> 32) -> ReLU -> LayerNorm`
- **Fusion Layer:** Concatenates `z_repr` and `z_chem` (total dim 64) and maps to 1 using a `Linear` layer.

## 2. Parameter Counts
- **Ordinary Transformer Baseline:** 85,761 parameters
- **Architecture-Control Model:** 90,049 parameters
- **Specialized Two-Branch Model:** 90,049 parameters
- **Parameter Increase:** +5.0% relative to the baseline.

## 3. Loss Functions and Objectives
The specialized model is trained with a multi-objective loss:
$$ L_{total} = L_{property} + \lambda_{repr} L_{invariance} + \lambda_{chem} L_{chemistry} + \lambda_{div} L_{diversity} $$

- **Property Regression Loss ($L_{property}$):** Mean Squared Error (MSE) on the scaled Bandgap target for both clean and valid randomized SMILES.
- **Representation-Invariance Loss ($L_{invariance}$):** Normalized cosine distance between clean and randomized representations. Weight: `0.5`
- **Chemistry-Sensitivity Loss ($L_{chemistry}$):** Margin-based contrastive loss ($L_2$ distance) enforcing a minimum distance between clean and substituted (attack budget=1) representations on Branch B. Weight: `0.5`, Margin: `0.5`
- **Diversity / De-collapse Loss ($L_{diversity}$):** Absolute cosine similarity penalty between Branch A and Branch B to prevent identical representations. Weight: `0.1`

## 4. Training Pairs
Generated from the clean training split (2,946 total sources):
- **Representation-Equivalent Pairs:** 2,887 successfully parsed and validated randomized SMILES pairs.
- **Chemistry-Changing Pairs:** 1,599 valid single-token substitution pairs.

## 5. Branch Diagnostics and Collapse Check
Preliminary diagnostics show that the branches have differentiated without representation collapse:
- **Norms:** Branch A (7.15), Branch B (6.04) — Embeddings do not vanish.
- **Variances:** Branch A (0.038), Branch B (0.784) — Neither branch collapses to a single point.
- **Orthogonality:** Cosine similarity between A and B is extremely low (-0.004).
- **Randomization Distance:** Branch A (0.54) vs Branch B (3.82) — Branch A has a lower overall response to randomization.
- **Substitution Distance:** Branch A (0.29) vs Branch B (1.93) — Branch B responds more strongly to substitutions, although Branch A is actually more invariant to substitution than randomization.

Phase 2 demonstrates branch differentiation without representation collapse. Semantic specialization remains to be tested on frozen unseen perturbation banks.

## 6. Scientific Limitations
- Branch diagnostics are preliminary evaluations on the training set. True adversarial robustness and specialization requires testing on frozen, unseen candidate banks in Phase 3.
- Exposed test metrics are strictly a reference point and should not be used for any hyperparameter tuning or early stopping.

## 7. Status
Phase 2 is successfully complete. The training is stable, representations do not collapse, and preliminary diagnostics confirm branch differentiation. It is safe to proceed to frozen candidate-bank evaluation.
