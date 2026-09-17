# Mathematical Attack Formulation and Framework

## 1. General Optimization Objective

Given a clean polymer sequence $x \in \mathcal{X}$, ground-truth band gap $y \in \mathbb{R}$ (eV), and trained surrogate model $f_\theta: \mathcal{X} \to \mathbb{R}$, an adversarial attack solves the constrained optimization problem:

$$\max_{x' \in \mathcal{S}(x)} \mathcal{L}_{\text{adv}}\left(f_\theta(x'), f_\theta(x), y\right) \quad \text{subject to} \quad x' \in \mathcal{C}(x)$$

where:
- $x$ is the original, unperturbed PSMILES sequence string.
- $x'$ is the adversarial candidate PSMILES sequence string.
- $\mathcal{S}(x)$ is the space of valid sequence perturbations accessible from $x$.
- $f_\theta(x)$ is the surrogate model prediction for input $x$ in eV.
- $\mathcal{L}_{\text{adv}}$ is the adversarial loss objective.
- $\mathcal{C}(x)$ is the domain constraint set enforcing chemical plausibility.

---

## 2. Attack Loss Objectives ($\mathcal{L}_{\text{adv}}$)

The framework supports three specific loss objectives depending on attacker intent:

### A. Untargeted Prediction Drift ($\mathcal{L}_{\text{drift}}$)
Measures the absolute shift in model prediction relative to the clean baseline prediction:
$$\mathcal{L}_{\text{drift}}(x') = |f_\theta(x') - f_\theta(x)|$$

### B. Error Maximization Loss ($\mathcal{L}_{\text{err}}$)
Maximizes prediction error relative to the ground-truth band gap $y$:
$$\mathcal{L}_{\text{err}}(x') = \left(f_\theta(x') - y\right)^2$$

### C. Targeted Shift Loss ($\mathcal{L}_{\text{target}}$)
Forces the prediction toward a specific target bandgap value $y_{\text{target}}$:
$$\mathcal{L}_{\text{target}}(x') = -\left(f_\theta(x') - y_{\text{target}}\right)^2$$

---

## 3. Distance Metrics and Constraint Formulation

### A. Tanimoto Structural Similarity ($S_{\text{Tanimoto}}$)
Calculated using Morgan circular fingerprints (radius $= 2$, $2048$ bits):
$$S_{\text{Tanimoto}}(x, x') = \frac{|\mathbf{fp}(x) \cap \mathbf{fp}(x')|}{|\mathbf{fp}(x) \cup \mathbf{fp}(x')|}$$
Constraint: $S_{\text{Tanimoto}}(x, x') \ge 0.5$.

### B. Sequence Edit Distance ($d_{\text{edit}}$)
Measures the number of atomic sequence modification operations (substitutions, insertions, deletions) applied to transform $x$ into $x'$:
$$d_{\text{edit}}(x, x') \le k_{\text{max}}$$
where default edit budget $k_{\text{max}} = 2$.

---

## 4. Discrete Modification Operators

The attack generator modifies PSMILES sequences via discrete operators:

1. **Substitution Operator** ($\mathcal{O}_{\text{sub}}$): Replaces an atom/token at position $i$ with a compatible token $t' \in \mathcal{V}$ (role-preserving: atom $\to$ atom).
2. **Insertion Operator** ($\mathcal{O}_{\text{ins}}$): Inserts a valid token $t' \in \mathcal{V}$ into non-structural positions.
3. **Deletion Operator** ($\mathcal{O}_{\text{del}}$): Removes a non-structural token while preserving ring closure and branch parenthesis balance.
4. **Bioisosteric Swap Operator** ($\mathcal{O}_{\text{bio}}$): Replaces functional group tokens according to chemical bioisosteric matrices ($-\text{F} \leftrightarrow -\text{Cl}$, $-\text{OH} \leftrightarrow -\text{SH}$).
