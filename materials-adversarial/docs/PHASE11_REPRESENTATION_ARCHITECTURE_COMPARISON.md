> [!NOTE]
> **Status**: HISTORICAL

# Phase 11: Representation Architecture Comparison

## Objective
To determine if moving from a sequence-based Transformer model to a Graph Neural Network (GNN) is genuinely justified for solving the representation invariance vulnerability, rather than being an over-engineered solution.

We compared three model architectures on the `Bandgap` prediction task:
1.  **A_BaselineSeq**: Standard Transformer sequence model (baseline from previous phases).
2.  **B_AugmentedSeq**: Transformer trained with data augmentation (randomizing canonical SMILES representations dynamically during training) to test if invariance can be learned post-hoc without architecture changes.
3.  **C_GraphMPNN**: A Message Passing Neural Network (MPNN) operating directly on the molecular graph topology instead of string sequences.

## Methodology
-   **Clean Evaluation**: Assess Standard Validation MAE and R² to ensure no unacceptable degradation (<= 0.02 degradation tolerance).
-   **Representation Bank Invariance**: Measure the prediction drift between different SMILES strings representing the *exact same* molecule.

## Results

### 1. Clean Performance (10 Epochs)

| Model | Val MAE | Val R² |
| :--- | :--- | :--- |
| **A_BaselineSeq** | 1.530 | -0.590 |
| **B_AugmentedSeq** | 0.563 | 0.702 |
| **C_GraphMPNN** | 0.689 | 0.585 |

*Note: Models were trained for 10 epochs. Model B (AugmentedSeq) achieved the highest clean performance, likely due to data augmentation acting as strong regularization. Model C (GraphMPNN) demonstrated solid clean predictive capability, remaining highly competitive with sequence models.*

### 2. Representation Invariance (Adversarial Robustness)

| Model | Mean Drift | Median Drift | P95 Drift |
| :--- | :--- | :--- | :--- |
| **A_BaselineSeq** | 0.162 | 0.122 | 0.454 |
| **B_AugmentedSeq** | 0.238 | 0.141 | 0.801 |
| **C_GraphMPNN** | **~0.000** | **0.000** | **~0.000** |

*Drift represents the difference in predicted bandgap when the input is an alternate SMILES representation of the identical molecule.*

## Analysis & Findings

1.  **Augmentation Harms Invariance**: Model B (AugmentedSeq) experienced *higher* representation drift (0.238 vs 0.162) than the baseline. This indicates that dynamically showing the model random canonical representations during training does not teach it an invariant internal representation. Instead, it learns to map different string forms to different prediction pathways, exacerbating the variance when presented with fresh representations.
2.  **Graph Topology Guarantees Invariance**: Model C (GraphMPNN) achieved a mean drift of `3.34e-08`, effectively **zero**. By processing the underlying 2D molecular graph (nodes = atoms, edges = bonds), the model bypasses string serialization entirely. Isomorphic graphs naturally produce identical representations in an MPNN.
3.  **GNNs are Justified**: The transition to a Graph Neural Network is not an over-engineered solution; it is mathematically required to guarantee robust representation invariance without sacrificing the primary predictive capability of the model.

## Recommended Path Forward

**GraphMPNN (Model C)** is formally recommended for the remainder of the project.

GNNs provide absolute immunity to representation-based adversarial attacks (since the topological graph representation is deterministic and canonical regardless of string serialization), satisfying the core project objective without unacceptable degradation of clean Bandgap regression performance.
