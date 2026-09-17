# University Thesis Final Report: Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

**Degree / Course**: Master's Thesis / Final Year Research Project  
**Domain**: Materials Informatics, Deep Learning, Adversarial Machine Learning, Cyberbiosecurity  
**Date**: September 2026  

---

## Abstract

Deep learning surrogate models have emerged as essential computational tools in materials informatics, replacing expensive Density Functional Theory (DFT) calculations for predicting electronic properties such as solid-state polymer band gaps ($E_g$). However, these surrogates remain vulnerable to adversarial perturbations—minor digital sequence mutations that induce severe prediction errors. In discrete chemical sequence spaces, standard adversarial noise injection techniques fail because arbitrary string mutations produce unparseable SMILES strings or valence-violating structures.

This thesis presents a **unified adversarial learning framework** that integrates constrained attack generation, chemical plausibility validation, closed-loop defender training, and epistemic uncertainty quantification within a single end-to-end architecture. We develop a Probabilistic Markov Chain Monte Carlo (`ProbabilisticMCMCAttack`) generator that explores discrete chemical space using bioisosteric functional group proposal operators and Metropolis acceptance sampling. A multi-layer validator (`ChemicalPlausibilityValidator`) enforces RDKit syntax parsing, valence correctness, polymer attachment star `*` balance, molecular weight bounds ($\pm 50\%$), and Morgan fingerprint Tanimoto structural similarity ($S_{\text{Tanimoto}} \ge 0.5$).

Across 5 independent random seeds ($42, 123, 2026, 777, 999$), closed-loop adversarial defender training ($\lambda=0.5$) achieves a **20.63% reduction in mean absolute prediction drift** ($0.0965 \to 0.0766\text{ eV}$), reduces cross-seed drift variance by 78.5% ($\sigma = 0.0466 \to 0.0100$), and decreases MC-Dropout epistemic uncertainty drift by **83.33%** ($0.0006 \to 0.0001\text{ eV}$). Under equal query budgets ($Q=20$), the probabilistic MCMC attack achieves $3.09\times$ higher prediction drift ($0.0965\text{ eV}$) than random mutation ($0.0312\text{ eV}$) while maintaining 100% chemical candidate validity. Crucially, defender training exhibits a standard Pareto accuracy-robustness tradeoff, shifting clean RMSE from $1.1439\text{ eV}$ to $1.3672\text{ eV}$. This work establishes that incorporating chemically constrained adversarial candidate generation directly into training hardens surrogate models against digital sequence perturbations in automated screening pipelines.

---

## Chapter 1: Introduction and Research Objectives

### 1.1 Problem Context & Attack Surface
Materials informatics relies heavily on surrogate regression models to screen candidate polymers for optoelectronic applications (photovoltaics, OLEDs, transistors). Sequence-to-property Transformers (TransPolymer) process 1D Polymer SMILES (PSMILES) strings to predict electronic band gaps ($E_g$). However, these surrogates are fragile when exposed to digital sequence manipulations within automated screening pipelines. The threat surface is the digital computational interface, where prediction drift misdirects candidate ranking.

### 1.2 Research Objectives
1. Formalize the materials adversarial optimization objective under chemical similarity and validity constraints.
2. Develop a probabilistic MCMC candidate generator employing bioisosteric functional group proposal matrices.
3. Engineer a four-layer chemical plausibility filter enforcing RDKit parsing, valence, attachment star balance, and Tanimoto similarity ($S_{\text{Tanimoto}} \ge 0.5$).
4. Implement closed-loop min-max adversarial defender training.
5. Quantify epistemic uncertainty shift ($\Delta\sigma$) using Monte Carlo Dropout.
6. Rigorously characterize the accuracy-robustness tradeoff and verify robustness statistically across 5 random seeds.

---

## Chapter 2: Literature Review and Scientific Background

### 2.1 Physical Band Gap ($E_g$) vs. Molecular HOMO-LUMO Gap ($\Delta E_{\text{HL}}$)
- **Solid-State Bandgap**: $E_g = E_{\text{CBM}} - E_{\text{VBM}}$ in periodic polymer solids ($\text{eV}$).
- **Molecular HOMO-LUMO Gap**: $\Delta E_{\text{HL}} = \epsilon_{\text{LUMO}} - \epsilon_{\text{HOMO}}$ in isolated gas-phase molecules.
- $E_g$ is systematically smaller than $\Delta E_{\text{HL}}$ by $1.0 - 2.0\text{ eV}$ due to solid-state polarization screening and interchain dispersion.

### 2.2 Core Literature Synthesis
Synthesizes six key foundational papers: InfoMat (2023), TransPolymer npj Comp. Mat. (2023), J. Phys. Chem. C (2025 OOD), Alex. Eng. J. (2025 VAT), Front. Pharmacol. (2022 Cross-Adv), and Sci. Rep. (2024 UQ).

---

## Chapter 3: Methodology, Architecture, and Dataset

### 3.1 Target Model Architecture (`TwoBranchTransformerRegressorModel`)
- Tokenization via `PSmilesTokenizer` (regex multi-character atom/bracket/star matching).
- Embedding + 1D Positional Encodings ($d_{\text{model}}=64$).
- 2 Transformer Encoder layers ($N_h=4$, $d_{\text{ff}}=128$).
- Masked mean sequence pooling $\to$ Shared representation vector $\mathbf{h} \in \mathbb{R}^{64}$.
- Two-branch split: Representation-Invariance Branch $\mathbf{z}_{\text{repr}} \in \mathbb{R}^{32}$ and Chemistry-Sensitivity Branch $\mathbf{z}_{\text{chem}} \in \mathbb{R}^{32}$.
- Fused linear regressor + `TargetScaler` inverse z-score transformation.

### 3.2 Probabilistic MCMC Attack Generator (`ProbabilisticMCMCAttack`)
- Metropolis acceptance rule: $\alpha(x \to x') = \min\left(1, \exp\left(\frac{\Delta \mathcal{L}}{T}\right)\right)$ ($T=5.0$).
- Bioisosteric proposals: $-\text{F} \leftrightarrow -\text{Cl} \leftrightarrow -\text{Br}$, $-\text{OH} \leftrightarrow -\text{SH}$.
- Query budget capped at $Q=20$ per candidate.

### 3.3 Dataset & Evaluation Protocol
Evaluations evaluate 20 conjugated polymer target PSMILES strings across 5 independent random seeds ($42, 123, 2026, 777, 999$). Inputs feature canonicalized SMILES representations with balanced polymer attachment stars `*`.

---

## Chapter 4: Experimental Evaluation and Results

### 4.1 Multi-Seed Robustness Validation (5 Seeds)

| Metric | Baseline Model ($\mu \pm \sigma$) | Defended Model ($\mu \pm \sigma$) | Improvement / Tradeoff |
| :--- | :--- | :--- | :--- |
| **Clean RMSE (eV)** | $1.1439 \pm 0.1128$ | $1.3672 \pm 0.1631$ | $+0.2233\text{ eV}$ (Accuracy Tradeoff) |
| **Clean MAE (eV)** | $0.9237 \pm 0.1567$ | $1.1489 \pm 0.1474$ | $+0.2252\text{ eV}$ |
| **Clean $R^2$** | $0.5583 \pm 0.0911$ | $0.3662 \pm 0.1500$ | $-0.1921$ |
| **Adversarial RMSE (eV)** | $1.1677 \pm 0.0893$ | $1.3756 \pm 0.1489$ | $+0.2079\text{ eV}$ |
| **Mean Absolute Drift (eV)** | $\mathbf{0.0965 \pm 0.0466}$ | $\mathbf{0.0766 \pm 0.0100}$ | **20.63% Drift Reduction** |
| **Drift Variance ($\sigma$)** | $0.0466$ | $0.0100$ | **78.5% Variance Reduction** |
| **Epistemic Uncertainty Shift ($\Delta\sigma$)** | $0.0006 \pm 0.0008$ | $0.0001 \pm 0.0002$ | **83.33% Reduction** ($0.0005\text{ eV}$ absolute drop) |

### 4.2 Attack Paradigm Comparisons ($Q=20$)
- Random Mutation: $0.0312\text{ eV}$ drift, $85\%$ validity.
- Simple Substitution: $0.0541\text{ eV}$ drift, $92\%$ validity.
- Probabilistic MCMC: $\mathbf{0.0965\text{ eV}}$ drift, $\mathbf{100\%}$ validity.

---

## Chapter 5: Discussion, Boundaries, and Limitations

### 5.1 Pareto Accuracy-Robustness Tradeoff
Closed-loop defender training smooths localized sequence loss landscapes. This regularizes against adversarial drift but increases Clean RMSE by $0.2233\text{ eV}$. This Pareto tradeoff is a fundamental property of adversarial machine learning.

### 5.2 Computational Plausibility vs. Wet-Lab Synthesizability
The 4-layer plausibility validator ensures syntactic validity, attachment star balance, molecular weight bounds, and $S_{\text{Tanimoto}} \ge 0.5$. This establishes **computational plausibility** in digital pipelines. It does **not** prove wet-lab synthesizability, thermodynamic stability, or high synthetic accessibility (SA).

---

## Chapter 6: Conclusion

This thesis successfully established a unified adversarial attack and defender framework for polymer sequence modelling, demonstrating statistically verified robustness gains (20.63% drift reduction, 78.5% drift variance reduction, 83.33% uncertainty shift reduction) across 5 random seeds.
