# Threat Model: Adversarial Attack Surfaces in Materials Sequence Modelling

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

## 1. Threat Model Overview

The threat model defines the security assumptions, attacker capabilities, knowledge levels, query constraints, success criteria, and physical vs. digital boundaries for adversarial attacks against machine learning surrogate models predicting polymer properties.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    THREAT MODEL MATRIX                                  │
├───────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Attacker Objective        │ Maximize bandgap prediction drift |f_theta(x') - f_theta(x)|│
│ Threat Surface Scope      │ Automated digital screening pipelines & web APIs            │
│ Knowledge Access          │ Black-box query access to predictor score output            │
│ Query Budget Constraint   │ Q <= 20 candidate model evaluations per source sample        │
│ Domain Constraints        │ Valid RDKit syntax, valence, MW bounds, S_Tanimoto >= 0.5   │
│ Success Threshold         │ Prediction drift |f_theta(x') - f_theta(x)| > 0.5 eV        │
└───────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 2. Attacker Profile and Capabilities

### A. Attacker Intent and Objective
The attacker seeks to manipulate a sequence-to-property surrogate model $f_\theta: \mathcal{X} \to \mathbb{R}$ into emitting incorrect solid-state polymer band gap predictions ($E_g$ in eV). The attacker operates in two primary modes:
1. **Untargeted Prediction Drift**: Maximize prediction shift $|f_\theta(x') - f_\theta(x)|$ relative to the clean baseline.
2. **Targeted Shift / Error Maximization**: Force the model to predict a target energy value $y_{\text{target}}$ or maximize absolute prediction error $(f_\theta(x') - y)^2$ vs ground truth $y$.

### B. Attacker Knowledge Level
- **Black-Box Access (Query Only)**: The attacker has no access to model parameters $\theta$, gradient vectors $\nabla_\theta f$, architecture weights, or training data splits.
- **Score Output Access**: The attacker can query the model with valid PSMILES strings $x'$ and receive the predicted scalar property value $f_\theta(x')$ in eV.

### C. Query Budget Constraint ($Q$)
- The attacker is constrained to a fixed maximum query budget of $Q = 20$ forward predictions per source molecule.
- This reflects realistic API rate limits, computational cost bounds, or anomaly detection thresholds in automated materials discovery platforms.

---

## 3. Attack Surfaces in Materials Informatics

Adversarial attack surfaces emerge at multiple stages of digital materials pipelines:

1. **Digital Sequence Submissions**: Web APIs accepting SMILES/PSMILES strings for virtual screening or high-throughput property estimation.
2. **Automated Candidate Selection**: Machine learning surrogates feeding candidate lists into automated computational DFT screening loops. An adversarial prediction shift misdirects resource allocation by prioritizing suboptimal polymer candidates.
3. **Database & Literature Ingestion**: Material databases ingesting unverified user-contributed PSMILES sequence strings.

---

## 4. Attacker Constraints & Scientific Plausibility

Unlike computer vision attacks where $\ell_p$-norm bounded noise ($\|x' - x\|_\infty \le \epsilon$) is added to continuous pixel grids, chemical sequence perturbations must obey strict domain rules:

$$\mathcal{C}(x) = \left\{ x' \in \mathcal{X} \;\middle|\; \begin{array}{l} \text{ValidChemistry}(x') = \text{True}, \\ \text{Count}_*(x') == \text{Count}_*(x), \\ 0.5 \le \frac{MW(x')}{MW(x)} \le 1.5, \\ S_{\text{Tanimoto}}(x, x') \ge 0.5 \end{array} \right\}$$

If a perturbation breaks RDKit parsing, alters explicit valence, drops polymer attachment stars `*`, or drastically changes molecular weight ($\Delta MW > 50\text{ g/mol}$), it is rejected.

> [!NOTE]
> **Boundary of Realism**: These constraints enforce **computational plausibility** in sequence space, ensuring candidates are valid, parseable chemical SMILES strings. They do **not** guarantee wet-lab synthetic accessibility (SA score) or physical thermodynamic stability.

---

## 5. Attacker vs. Defender Game Formulation

The interaction is formalized as a zero-sum game between Attacker $\mathcal{A}$ and Defender $\mathcal{D}$:
- **Attacker Goal**: $\max_{x' \in \mathcal{C}(x)} |f_\theta(x') - f_\theta(x)|$.
- **Defender Goal**: $\min_{\theta} \mathbb{E}_{(x, y)} \left[ (1-\lambda) \mathcal{L}(f_\theta(x), y) + \lambda \mathcal{L}(f_\theta(x_{\text{adv}}'), y) \right]$ with $\lambda = 0.5$.
