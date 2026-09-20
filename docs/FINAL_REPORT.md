# FINAL RESEARCH REPORT
## Adversarial Robustness in Polymer Property Prediction

### 1. Abstract and Motivation
Machine learning models applied to materials science, particularly Transformer-based architectures operating on SMILES-like string representations (PSMILES), exhibit high predictive accuracy for properties such as the bandgap ($E_g$). However, their reliability under adversarial distribution shifts remains poorly understood. This research evaluates the robustness of a Transformer model against two distinct classes of adversarial perturbations:
1. **Representation-Preserving Perturbations**: Syntactically valid but chemically identical string serializations.
2. **Chemistry-Changing Perturbations**: Syntactically and heuristically plausible local edits to the molecular graph (MCMC edits).

### 2. Threat Model and Validation
The threat model incorporates a rigorous validation pipeline that enforces syntactic correctness (via RDKit), attachment-point preservation, branching/ring integrity, molecular weight bounds, and structural similarity (Tanimoto constraints). The evaluated attacks strictly abide by these heuristics, ensuring that identified vulnerabilities represent genuine flaws in the model's structural reasoning rather than parsing edge-cases.

### 3. Key Findings

#### 3.1 Representation Sensitivity
The baseline Transformer model demonstrates severe sensitivity to chemically identical serializations. Randomizing the SMILES representation of the same molecule yields a mean prediction drift of 0.897 eV on the scaffold split, indicating that the model relies heavily on serialization artifacts.

#### 3.2 Efficacy of Data Augmentation
Training with Randomized SMILES Augmentation dramatically reduces this serialization vulnerability, suppressing representation drift to 0.408 eV. Furthermore, it improvides evidence clean out-of-distribution (Scaffold Split) RMSE from 0.700 eV to 0.663 eV, proving that representation robustness acts as a beneficial structural regularizer.

#### 3.3 Sequence Length as a Predictive Proxy
Statistical analyses indicate a correlation between true bandgap and sequence length ($r \approx -0.11$) but a much stronger association between the model's absolute error and sequence length. Crucially, adversarial prediction drift is positively associated with the number of tokens inserted or deleted, consistent with the hypothesis that the network relies on sequence length as a proxy shortcut for structural complexity.

#### 3.4 Cross-Attack Generalization
Defending against representation-preserving attacks (RandSmiles) and chemistry-changing attacks (MCMC) addresses orthogonal vulnerabilities. MCMC defense barely reduces Random-SMILES drift, and vice-versa. However, a **Combined Defense** strategy performs best across the evaluated metrics/configurations, reducing both MCMC drift (to 0.182 eV) and Random-SMILES drift (to 0.378 eV).

### 4. Limitations and Future Work
**DFT Physical Oracle**: The primary limitation of this work is the absence of Density Functional Theory (DFT) calculations to verify the true bandgap of the chemistry-changing MCMC adversarial edits. Consequently, while the MCMC attacks successfully expose the model's extreme sensitivity to constrained chemical edits, we cannot definitively claim that the edited molecules retain their identical physical bandgap. The measured defense improvements reflect *regularization consistency* rather than guaranteed physical accuracy.

### 5. Conclusion
Transformer models for sequence-based materials property prediction are highly susceptible to both representation instability and localized structural perturbations. Robustness requires a multi-faceted defense strategy that simultaneously normalizes serialization invariance and smooths local topological sensitivity.
