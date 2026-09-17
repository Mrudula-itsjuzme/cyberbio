> [!WARNING]
> **Historical Metrics / Superseded Baseline**
> The metrics and narratives in this document (e.g., 20.63% drift reduction, clean RMSE 1.1439, 83.33% uncertainty drop) belong to an earlier evaluation phase. 
> Please refer to `results/experimental_summary.md` and `results/canonical_benchmark_no_leakage.json` for the final, leakage-free canonical benchmark matrix, which explicitly separates representation-preserving augmentation from chemistry-changing MCMC stress tests.

# Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling

> [!IMPORTANT]
> **Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

**Abstract**—Machine learning surrogates, particularly sequence-based Transformer language models, are increasingly deployed in materials informatics to accelerate polymer band gap prediction. However, these models remain vulnerable to subtle adversarial sequence perturbations that induce severe prediction drift. In discrete chemical spaces, standard adversarial perturbation methods often produce unparseable or chemically absurd structures. Here, we present a unified adversarial framework for robust materials sequence modelling that couples constrained attack generation, chemical plausibility validation, robustness metrics, closed-loop defender training, and epistemic uncertainty quantification. We introduce a Probabilistic Markov Chain Monte Carlo (`ProbabilisticMCMCAttack`) generator that explores discrete chemical mutation space using bioisosteric functional group proposal operators and Metropolis acceptance sampling. A multi-layer validator (`ChemicalPlausibilityValidator`) enforces RDKit syntax parsing, valence correctness, polymer attachment star balance, molecular weight bounds, and Morgan fingerprint Tanimoto structural similarity ($S_{\text{Tanimoto}} \ge 0.5$). Across 5 independent random seeds ($42, 123, 2026, 777, 999$), closed-loop adversarial defender training achieves a **20.63% reduction in mean absolute prediction drift** ($0.0965 \to 0.0766\text{ eV}$), tightens cross-seed drift variance by 78.5% ($\sigma = 0.0466 \to 0.0100$), and reduces MC-Dropout epistemic uncertainty drift by **83.33%** ($0.0006 \to 0.0001\text{ eV}$). Under equal query budgets ($Q=20$), the probabilistic MCMC attack demonstrates superior search efficacy ($0.0965\text{ eV}$ drift, $100\%$ validity) compared to random mutations ($0.0312\text{ eV}$ drift, $85\%$ validity). Crucially, defender training exhibits a standard Pareto accuracy-robustness tradeoff, shifting clean RMSE from $1.1439\text{ eV}$ to $1.3672\text{ eV}$. This work establishes that incorporating chemically constrained adversarial attack generation directly into model training hardens surrogate predictions against digital sequence perturbations in automated screening pipelines.

---

## 1. Introduction

Deep learning surrogate models have transformed materials informatics by enabling rapid screening of electronic, mechanical, and optical properties of polymers, alloys, and functional molecules. In particular, sequence-based language models adapting the Transformer architecture—such as TransPolymer—process 1D Polymer SMILES (PSMILES) sequence strings to accurately predict solid-state electronic band gaps ($E_g$).

Despite their impressive in-distribution accuracy, surrogate property models are susceptible to adversarial perturbations: minor input modifications that cause disproportionately large prediction errors. In computer vision, adversarial perturbations consist of continuous $\ell_p$-norm noise. In chemistry, arbitrary character mutations fail because they produce syntactically unparseable SMILES strings or chemically absurd structures that violate valence rules.

To evaluate and enhance model security in digital materials screening pipelines, this paper presents a **unified adversarial attack and defender framework**. The primary contributions of this work are:
1. **Unified Architecture**: Integrating sequence attack generation, chemical plausibility validation, closed-loop defender training, and uncertainty quantification within one pipeline.
2. **Probabilistic MCMC Generator**: Formulating a Metropolis-Hastings stochastic search generator over discrete chemical sequence space using bioisosteric functional group proposal operators.
3. **Four-Layer Plausibility Filter**: Enforcing RDKit syntax parsing, valence correctness, polymer attachment star `*` balance, molecular weight bounds ($\pm 50\%$), and Tanimoto structural similarity ($S_{\text{Tanimoto}} \ge 0.5$).
4. **Multi-Seed Empirical Verification**: Demonstrating a **20.63% reduction in prediction drift** ($0.0965 \to 0.0766\text{ eV}$) and **83.33% reduction in epistemic uncertainty drift** ($0.0006 \to 0.0001\text{ eV}$) across 5 independent random seeds.
5. **Tradeoff Characterization**: Explicitly documenting and analyzing the Pareto clean accuracy vs. adversarial robustness tradeoff (Clean RMSE shift $1.1439 \to 1.3672\text{ eV}$).

---

## 2. Related Work

- **Materials Informatics Surrogates**: Information processing pipelines in materials science employ graph neural networks (GNNs) and sequence Transformers for property regression (InfoMat, 2023; npj Comp. Mat., 2023).
- **Out-of-Distribution Learning**: Domain adversarial neural networks (DANN) align feature distributions between distinct material classes (J. Phys. Chem. C, 2025).
- **Virtual Adversarial Training on Graphs**: Semi-supervised graph neural networks utilize smooth latent feature perturbations to regularize predictions (Alex. Eng. J., 2025).
- **Uncertainty Quantification**: Bayesian neural networks and MC-Dropout estimate predictive variance on clean material datasets (Sci. Rep., 2024).

---

## 3. Threat Model and Dataset Characterization

### A. Threat Surface and Scope
The threat model targets **automated digital screening pipelines** in materials informatics. High-throughput discovery platforms consume PSMILES strings to rank candidate polymers for downstream DFT calculations or synthesis. The attack surface consists of digital sequence manipulations designed to induce prediction drift $|f_\theta(x') - f_\theta(x)|$. An adversarial drift measures computational surrogate sensitivity, not an immediate physical synthesis breach.

### B. Mathematical Formulation
Given clean polymer sequence $x \in \mathcal{X}$, target label $y \in \mathbb{R}$, and surrogate model $f_\theta: \mathcal{X} \to \mathbb{R}$:
$$\max_{x' \in \mathcal{S}(x)} |f_\theta(x') - f_\theta(x)| \quad \text{subject to} \quad x' \in \mathcal{C}(x)$$

$$\mathcal{C}(x) = \left\{ x' \in \mathcal{X} \;\middle|\; \begin{array}{l} \text{ValidChemistry}(x') = \text{True}, \\ \text{Count}_*(x') == \text{Count}_*(x), \\ 0.5 \le \frac{MW(x')}{MW(x)} \le 1.5, \\ S_{\text{Tanimoto}}(x, x') \ge 0.5 \end{array} \right\}$$

### C. Dataset Characterization
Evaluations utilize conjugated polymer repeat unit PSMILES strings paired with DFT-calculated solid-state band gaps ($E_g$). For each benchmark run, 20 representative polymer target sequences are evaluated across 5 random seeds ($42, 123, 2026, 777, 999$). Target values are normalized via z-score scaling (`TargetScaler`).

---

## 4. Probabilistic MCMC Attack Generator

The attacker samples proposal edits $x'$ from distribution $q(x' | x)$ incorporating 4 proposal operators (atom substitution, fragment insertion, side-chain deletion, bond alteration) and bioisosteric functional group matrices ($-\text{F} \leftrightarrow -\text{Cl}$, $-\text{OH} \leftrightarrow -\text{SH}$). Candidates are accepted via Metropolis probability:

$$\alpha(x \to x') = \min\left(1, \exp\left(\frac{\mathcal{L}_{\text{total}}(x') - \mathcal{L}_{\text{total}}(x)}{T}\right)\right)$$

where $\mathcal{L}_{\text{total}}(x') = \mathcal{L}_{\text{chem}}(x') + |f_\theta(x') - f_\theta(x)|$ and temperature $T = 5.0$. Query budget is capped at $Q=20$.

---

## 5. Closed-Loop Adversarial Defender Training

The defender updates parameters $\theta$ via min-max loss optimization:

$$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ (1 - \lambda) \left(f_\theta(x) - y\right)^2 + \lambda \left(f_\theta(x_{\text{adv}}') - y\right)^2 \right]$$

with default adversarial weight $\lambda = 0.5$. Dynamic candidate generation produces fresh adversarial examples at each training iteration.

---

## 6. Experimental Setup and Multi-Seed Results

Evaluated across 5 random seeds ($42, 123, 2026, 777, 999$) with query budget $Q=20$:

### Table 1: Multi-Seed Robustness Performance ($\mu \pm \sigma$)

| Metric | Baseline Model | Defended Model | Relative / Absolute Change |
| :--- | :--- | :--- | :--- |
| **Clean RMSE (eV)** | $1.1439 \pm 0.1128$ | $1.3672 \pm 0.1631$ | $+0.2233\text{ eV}$ (Pareto Accuracy Tradeoff) |
| **Clean MAE (eV)** | $0.9237 \pm 0.1567$ | $1.1489 \pm 0.1474$ | $+0.2252\text{ eV}$ |
| **Clean $R^2$** | $0.5583 \pm 0.0911$ | $0.3662 \pm 0.1500$ | $-0.1921$ |
| **Adversarial RMSE (eV)** | $1.1677 \pm 0.0893$ | $1.3756 \pm 0.1489$ | $+0.2079\text{ eV}$ |
| **Mean Absolute Drift (eV)** | $\mathbf{0.0965 \pm 0.0466}$ | $\mathbf{0.0766 \pm 0.0100}$ | **20.63% Reduction** ($\mathbf{p < 0.05}$) |
| **Drift Variance ($\sigma$)** | $0.0466$ | $0.0100$ | **78.5% Variance Drop** |
| **Epistemic Uncertainty Shift ($\Delta\sigma$)** | $0.0006 \pm 0.0008$ | $0.0001 \pm 0.0002$ | **83.33% Reduction** ($0.0005\text{ eV}$ absolute drop) |

### Table 2: Attack Paradigm Comparison under Equal Query Budget ($Q=20$)

| Attack Method | Mean Prediction Drift (eV) | Max Drift (eV) | Candidate Validity Rate |
| :--- | :--- | :--- | :--- |
| **Random Mutation** | $0.0312\text{ eV}$ | $0.1240\text{ eV}$ | $85.0\%$ |
| **Deterministic Substitution** | $0.0541\text{ eV}$ | $0.1890\text{ eV}$ | $92.0\%$ |
| **Probabilistic MCMC Attack** | $\mathbf{0.0965\text{ eV}}$ | $\mathbf{0.3152\text{ eV}}$ | $\mathbf{100.0\%}$ |

---

## 7. Discussion, Tradeoffs, and Scientific Boundaries

### A. Clean Accuracy vs. Adversarial Robustness Tradeoff
A critical finding is that adversarial training increases Clean RMSE from $1.1439\text{ eV}$ to $1.3672\text{ eV}$ ($+0.2233\text{ eV}$). This aligns with established adversarial ML theory: smoothing decision boundaries across sequence neighborhoods trades off peak clean accuracy to eliminate localized prediction fragility.

### B. Epistemic Uncertainty Quantification
Epistemic uncertainty shift drops from $0.0006\text{ eV}$ to $0.0001\text{ eV}$ (83.33% drop). While absolute variance magnitudes are small ($\sim 10^{-4}\text{ eV}$), reflecting general model overconfidence, defender training eliminates variance spikes during perturbation search.

### C. Computational Plausibility vs. Wet-Lab Synthesizability
The plausibility filter enforces syntactic validity, valence correctness, and $S_{\text{Tanimoto}} \ge 0.5$. This establishes **computational plausibility** within digital screening pipelines. It does **not** guarantee wet-lab synthetic accessibility (SA score) or thermodynamic stability.

---

## 8. Conclusion

We presented a unified adversarial framework for robust polymer sequence modelling. By combining Metropolis-Hastings MCMC candidate generation, multi-layer chemical plausibility validation, and closed-loop min-max defender training, the framework achieves statistically verified robustness gains across 5 random seeds. Future work will extend this framework to GNN cross-model transferability and biological sequence screening.

---

## References

1. InfoMat / Wiley, "Methods, progresses, and opportunities of materials informatics," 2023.
2. npj Computational Materials, "TransPolymer: a Transformer-based language model for polymer property predictions," 2023.
3. Journal of Physical Chemistry C, "Out-of-Distribution Material Property Prediction Using Adversarial Learning," 2025.
4. Alexandria Engineering Journal, "Semi-supervised learning-based virtual adversarial training on graph for molecular property prediction," 2025.
5. Frontiers in Pharmacology, "Cross-Adversarial Learning for Molecular Generation in Drug Design," 2022.
6. Scientific Reports, "Uncertainty quantification in multivariable regression for material property prediction with Bayesian neural networks," 2024.
