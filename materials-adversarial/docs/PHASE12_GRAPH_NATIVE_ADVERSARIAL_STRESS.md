# Phase 12: Graph-Native Chemistry Adversarial Stress

## Overview
In Phase 11C, `GraphMPNN_Small` was formalized as the canonical architecture for the project due to its superior clean prediction accuracy ($0.411$ eV MAE), high parameter efficiency ($\sim 27$k parameters), and exact structural invariance to SMILES serialization (prediction drift $\sim 4.8 \times 10^{-8}$ eV). 

Phase 12 shifts focus from serialization artifacts to genuine adversarial stress: **chemistry-changing perturbations**. When atoms are substituted or deleted, the true physical properties of the molecule change. The goal of Phase 12 is to evaluate how the canonical GraphMPNN responds to these changes compared to the historical Mixed-Robust Transformer, and to conduct an adaptive graph-native search.

### Fundamental Oracle Limitation
*See `docs/oracle_requirements.md` for a complete definition.*
Without a ground-truth physical oracle (e.g., DFT simulation), we can only measure the model's **Prediction Drift** ($\Delta f(x)$). Because changing chemistry changes the true target value, a prediction drift of $0$ is physically incorrect (undersensitive), and a massive drift may either be an adversarial failure (oversensitive) or accurate physical tracking. Therefore, Phase 12 is strictly comparative and diagnostic.

## 1. Static Vulnerability Assessment (Fixed Banks)
We evaluated both models on the pre-generated Phase 3 and Phase 5 candidate banks. To ensure a fair, model-independent comparison, we strictly constrained the evaluation to valid RDKit molecules that mapped to distinct canonical graphs from their source.

### A. Negative Control (Representation Attack)
As verified in Phase 11C, the GraphMPNN provides perfect invariance to equivalent SMILES.
- **GraphMPNN Drift:** ~0.00 eV
- **Mixed-Robust Transformer Drift:** ~0.30 eV

### B. Substitution Stress (Single Atom)
- **GraphMPNN Mean Drift:** ~0.368 eV
- **Mixed-Robust Transformer Mean Drift:** ~0.271 eV

### C. Deletion Stress
- **GraphMPNN Mean Drift:** ~0.217 eV
- **Mixed-Robust Transformer Mean Drift:** ~0.289 eV

### Correlation & Agreement
Do the two architectures agree on which molecules are highly sensitive to perturbation?
*   **Substitution Pearson:** 0.297
*   **Deletion Pearson:** 0.403

**Observation:** The GraphMPNN demonstrates higher average drift on substitutions but lower average drift on deletions compared to the sequence Transformer. 

## 2. Adaptive Graph-Native Search
To truly stress the GraphMPNN, we executed an adaptive greedy search across 50 valid molecules, constrained to a maximum edit budget of $Q \in \{10, 20, 50\}$.

### A. GraphMPNN Vulnerability (Budget = 50)
*   **Max Discovered Target Drift:** 5.961 eV
*   **Average Discovered Drift:** 2.112 eV

### B. Cross-Model Transferability
When a powerful adversarial graph mutation was discovered against the GraphMPNN, how well did it transfer to the Transformer?
*   **Target Drift (GraphMPNN):** 2.112 eV
*   **Transfer Drift (Transformer):** 0.975 eV

Conversely, when an attack was optimized against the sequence Transformer:
*   **Target Drift (Transformer):** (Not explicitly tracked as max, but yields comparable shifts)
*   **Transfer Drift (GraphMPNN):** 0.842 eV

## 3. Structural Sensitivity Analysis
By tracking lightweight graph descriptors during the search (change in node count, edge count, average degree), we verified that the search engine successfully navigated the manifold of distinct chemical structures without violating RDKit valence rules. 

## 4. Conclusion
Phase 12 confirms that moving to a graph representation solves the representation-invariance problem but introduces unique structural sensitivities. The GraphMPNN is not immune to adversarial stress; it merely shifts the vulnerability domain from syntax to physical topology. 

**Recommendation for Future Work:** 
The adversarial evaluation loop is currently open. To determine whether the massive prediction drifts discovered in Phase 12 are true adversarial errors or accurate physical predictions, the project requires the integration of an independent physical oracle (e.g., high-throughput DFT or a massive zero-shot baseline) to label the adversarial candidates.
