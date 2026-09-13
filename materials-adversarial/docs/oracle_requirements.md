# Oracle Limitations and the Missing Ground Truth

## 1. Introduction
In adversarial machine learning for physical sciences (like chemistry or materials science), a fundamental limitation arises when perturbing the input structure: the true physical property changes. In typical CV/NLP tasks, an adversarial perturbation (e.g., adding noise to an image of a cat) does not change the true label ("cat"). However, in chemistry, mutating a functional group or deleting an atom fundamentally alters the molecule, creating a novel entity whose true property (e.g., HOMO-LUMO gap) is unknown without experimental measurement or ab initio physical simulation.

## 2. The Adversarial "Drift" Metric
Because we lack a high-throughput physical oracle (like Density Functional Theory, DFT) to label thousands of adaptively generated adversarial candidates during the search process, we evaluate vulnerability strictly in terms of **Prediction Drift**:

$$ \Delta f(x) = |f_{model}(x_{adversarial}) - f_{model}(x_{original})| $$

While large prediction drifts are interesting, they are ambiguous:
*   **Case A (Over-sensitive):** The true physical property barely changed, but the model's prediction shifted by 1.0 eV. This is a true adversarial failure.
*   **Case B (Under-sensitive / Muted):** The true physical property changed by 1.0 eV, but the model's prediction remained perfectly invariant (drift = 0). This is *also* a failure, representing a physical "blind spot" in the model's representation.
*   **Case C (Accurate Tracking):** The model correctly predicts a massive shift because the chemistry genuinely changed drastically. This is desired behavior.

## 3. Requirement for Future Work
To resolve this ambiguity, future phases of research **MUST** introduce an independent ground-truth oracle. 

1.  **Surrogate Oracle (Weak):** A much larger, pre-trained zero-shot foundational model (e.g., a massive GNN or 3D conformer network) used strictly to estimate the physical bounds of the perturbation.
2.  **Simulation Oracle (Strong):** Automated Density Functional Theory (DFT) calculations on the subset of highly successful adversarial candidates.

**Rule:** The attacked model itself CANNOT supply ground truth. Claiming a model is robust simply because its chemistry drift is low is physically incorrect. Graph representations, which structurally mute minor perturbations by design, must be evaluated against physical truth to ensure they are not artificially under-sensitive.
