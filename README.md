# Cyberbio: Adversarial Robustness for Sequence Models
### *A unified DL + probabilistic framework for robust sequence learning and defense research*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository explores adversarial machine learning vulnerabilities, shortcut learning, and defense mechanisms in sequence models across two primary domains: **Materials Science** (polymer property prediction from PSMILES) and **Bio-Cybersecurity** (synthetic motif classification in genomic-like sequences).

The system is designed around a **closed-loop adversarial paradigm**: a probabilistic engine proposes plausible sequence modifications to probe a deep learning model's fragility, which are then used in adversarial training to harden the model.

---

## Academic Focus: Dual DL & Probability Mapping

This project explicitly bridges **Deep Learning** and **Probabilistic / Statistical Modeling**, making it suitable for evaluation in both disciplines:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PROBABILITY & STATISTICS                        │
│   • Bigram Transition Matrix P(tok_j | tok_i) with Laplace Smoothing   │
│   • Metropolis-style MCMC Sampling over Discrete Sequence Space        │
│   • Bayesian Epistemic Uncertainty via Monte Carlo Dropout (MC-Dropout)│
│   • Wilcoxon Signed-Rank Tests & 1,000-sample Bootstrap CIs            │
│   • Stochastic Sequence Generation & RandomSMILES Traversal Sampling   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Proposes candidate x'
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                             DEEP LEARNING                              │
│   • Multi-Head Transformer Encoder Regressor (TransPolymer, TwoBranch) │
│   • 1D Convolutional Neural Network (CNN_Distance & Motif Classifier)  │
│   • Denoising Sequence Autoencoder (Bidirectional GRU Encoder-Decoder) │
│   • Feature Attribution, Saliency & Shortcut Learning Analysis         │
│   • Adversarial Training & Consistency Loss Regularization             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Feedback / Drift: |f_θ(x') - f_θ(x)|
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           THE DEFENSE LOOP                             │
│   • Adversarial Data Augmentation (RandomSMILES + MCMC variants)       │
│   • Out-of-Distribution Generalization on Scaffold Splits              │
│   • Multi-Seed Robustness Validation (N=3, N=5)                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Deep Learning Dimension
* **Transformer Encoder Regressors:** Custom multi-head self-attention encoders with learned positional encodings, masked mean pooling, and MLP prediction heads predicting continuous physical properties ($E_g$ bandgap in eV, $T_g$ in K).
* **1D Convolutional Neural Networks:** PyTorch `Conv1d`, `MaxPool1d`, and dense layers classifying sequences from one-hot tensor representations `(4, L)`.
* **Deep Generative Sequence Models:** Bidirectional GRU encoder with GRU decoder trained via teacher-forced masked cross-entropy for sequence reconstruction.
* **Adversarial Deep Learning:** Formal adversarial training loops optimizing joint task loss and adversarial consistency penalties $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{consistency}}$.
* **Representation & Shortcut Learning:** Probing representation invariance vs. topological sensitivity and diagnosing model reliance on spurious correlations (e.g. sequence length or $k$-mer frequency).

### 2. Probability & Statistical Modeling Dimension
* **Discrete Markov Chains & Transition Matrices:** Bigram transition model $P(\text{token}_j \mid \text{token}_i)$ estimated from corpus data, row-normalized with **Laplace (add-$\alpha$) smoothing**, and sampled via categorical distributions.
* **Markov Chain Monte Carlo (MCMC) & Metropolis Sampling:** Stochastic search over discrete mutation space with proposal kernel $q(x' \mid x)$ and Metropolis acceptance probability:
  $$\alpha(x \to x') = \min\left(1, \exp\left(\frac{\Delta \mathcal{L}}{T}\right)\right)$$
  parameterized by temperature $T$ (simulated annealing), balancing exploration vs. drift exploitation.
* **Bayesian Deep Learning & Epistemic Uncertainty Quantification:** Monte Carlo Dropout (Gal & Ghahramani) sampling $N_{\text{mc}} = 15$ stochastic forward passes to compute:
  * Predictive Mean: $\hat{y}_{\text{mean}}(x) = \frac{1}{N_{\text{mc}}} \sum_{i=1}^{N_{\text{mc}}} f_{\theta^{(i)}}(x)$
  * Epistemic Variance: $\sigma^2(x) = \frac{1}{N_{\text{mc}}} \sum_{i=1}^{N_{\text{mc}}} (f_{\theta^{(i)}}(x) - \hat{y}_{\text{mean}}(x))^2$
  * Uncertainty Shift under Attack: $\Delta \sigma = \mathbb{E}[\sigma(x_{\text{adv}}) - \sigma(x_{\text{clean}})]$ (defended models achieve an **83.3% reduction** in uncertainty shift).
* **Statistical Inference & Hypothesis Testing:**
  * Non-parametric **Wilcoxon signed-rank tests** evaluating paired drift differences against null models.
  * **1,000-sample bootstrap confidence intervals** for effect sizes and attribution scores.
  * **Pearson ($r$)** and **Spearman ($\rho$)** correlation analyses for error attribution.
  * Multi-seed variance estimation reporting $(\mu \pm \sigma)$.

---

## Sub-Projects Overview

The repository is structured into two main sub-projects:

### 1. Materials Adversarial (`materials-adversarial/` & `Project/`)
**Target:** Predicting polymer properties directly from 1D sequence representations (Polymer SMILES / PSMILES).

> [!WARNING]
> **Scientific Status Update (2026-09-22): CONVERGENCE_FAILED**
> The physical oracle convergence plan for this project was halted on the local 15 GiB host because the required 12 Å vacuum regime exhausted practical memory limits (failed at 80/360 Ry on the 22-atom C/H smoke structure before SCF). 
> * No protocol freeze occurred.
> * No 12-polymer calibration or attack-pair DFT was run.
> * The finite capped-oligomer PBE HOMO-LUMO gap remains unvalidated against the historical periodic-chain target.
> The workflow is safely preserved and can be resumed on a 32–64 GiB host using `materials-adversarial/hpc/physical_oracle/`.

> [!IMPORTANT]
> **Lineage Disambiguation:**
> 1. **Historical Exploratory Lineage (OpenPoly $T_g$ in K):** Exploratory research on glass transition temperature using ~247 samples. Refuted the initial hypothesis that length-changing attacks are uniquely dangerous; proved global instability under token reordering and revealed shortcut learning confounds. *(See [`materials-adversarial/docs/HISTORICAL_TG_LINEAGE.md`](materials-adversarial/docs/HISTORICAL_TG_LINEAGE.md))*.
> 2. **Active Scientific Lineage (polyVERSE Bandgap $E_g$ in eV):** Current active research using 4,209 high-fidelity DFT polymer records (split 2,946 train / 631 val / 632 sealed test). Clean Test MAE: **0.4619 eV** ($R^2 = 0.8019$). Evaluates representation-preserving attacks (RandomSMILES) vs. chemistry-changing attacks (MCMC localized edits).

### 2. Bio-Cyber Adversarial (`bio-cyber-adversarial/`)
**Target:** Safe computational benchmark for investigating adversarial attacks on sequence-based biological classifiers.

* **Synthetic Benchmark:** Safe, strictly computational environment with zero real pathogenic or functional motifs.
* **Dataset Generation:** 20,000 synthetic DNA sequences ($L=50$ to $L=100$, alphabet $\{A, C, G, T\}$) with implanted synthetic motifs.
* **Benchmark Evolution:**
  * **V1:** Diagnosed and documented $k$-mer frequency leakage.
  * **V2:** Diagnosed composite distance artifacts.
  * **V3 Benchmark:** Enforces non-overlapping, distance-spaced motifs with counterfactual sanity checks, reducing shallow compositional shortcuts.
* **Baseline Classifier:** 1D CNN with interpretability sanity checks verifying whether predictions rely on motifs or background artifacts.

---

## Repository Structure

```
DL_cyberbio/
├── materials-adversarial/         # Active Materials Science adversarial pipeline
│   ├── src/materials_adv/        # Modular source: models, attacks, training, oracle
│   ├── docs/                     # 100+ research reports, threat models, & audit notes
│   ├── hpc/                      # HPC bundle for DFT physical oracle handoff
│   └── scripts/                  # Final experiment runners & figure generators
├── bio-cyber-adversarial/         # Bio-cybersecurity synthetic benchmark
│   ├── src/                      # CNN model, data generator, attack & evaluate scripts
│   └── data/                     # Raw & processed synthetic sequence datasets
├── Project/                      # Two-phase build specification and phase1 implementation
│   ├── phase1/                   # Standalone TransitionModel, MCMCSampler, and baseline
│   └── project_spec_two_phase.md # Specification defining DL & probabilistic roles
├── docs/                         # Global cross-project audits, viva guides, & LaTeX specs
├── results/                      # Final aggregated CSVs, figures, and statistical JSONs
└── Materials_Adversarial_Detailed_Report.tex  # Comprehensive academic manuscript
```

---

## Key Experimental Results

1. **Representation Brittleness:** Standard Transformer sequence models exhibit severe prediction drift under semantically equivalent string representations (RandomSMILES), revealing that models learn sequence syntax rather than pure molecular physics.
2. **Orthogonality of Defenses:** Defending against representation drift (via RandomSMILES augmentation) does not defend against topological edits (MCMC), and vice versa. A **Combined Defense** is required to achieve holistic robustness.
3. **Out-of-Distribution Regularization:** Random representation augmentation acts as a structural regularizer, improving generalization performance on out-of-distribution **Scaffold Splits**.
4. **Epistemic Uncertainty Stabilization:** Adversarial training reduces epistemic uncertainty shift under attack by **83.3%**, preventing the model from producing overconfident erroneous predictions on out-of-distribution perturbations.
5. **Attribution Hypothesis Refutation:** Rigorous Wilcoxon testing paired with matched controls decisively refuted the hypothesis that adversarial edits preferentially target high-attribution positions over random positions.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Mrudula-itsjuzme/cyberbio.git
cd cyberbio

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r materials-adversarial/requirements.txt
# (Optional) For Bio-Cyber: pip install -r bio-cyber-adversarial/requirements.txt
```

### 2. Running Verification & Smoke Tests

Run the materials attack pipeline smoke test:
```bash
python materials-adversarial/scripts/run_final_experiments.py --smoke
```

Run test suites:
```bash
pytest materials-adversarial/tests
pytest bio-cyber-adversarial/tests
```

### 3. Reproducing Full Experiments & Figures

To execute the complete benchmark and generate all final tables, ablation matrices, and statistical artifacts:
```bash
python materials-adversarial/scripts/run_final_experiments.py --full
```
Outputs and metrics are saved to `results/final/` and `docs/figures/`.

---

## Documentation & Viva Guides

* **Academic Report:** [`Materials_Adversarial_Detailed_Report.tex`](Materials_Adversarial_Detailed_Report.tex) / [PDF](Materials_Adversarial_Detailed_Report.pdf)
* **Comprehensive Viva Guide:** [`materials-adversarial/docs/VIVA_GUIDE.md`](materials-adversarial/docs/VIVA_GUIDE.md) and [`docs/VIVA_QUESTIONS.md`](docs/VIVA_QUESTIONS.md)
* **MCMC Attack Formulation:** [`materials-adversarial/docs/MCMC_ATTACK.md`](materials-adversarial/docs/MCMC_ATTACK.md)
* **Epistemic Uncertainty (MC-Dropout):** [`materials-adversarial/docs/UNCERTAINTY.md`](materials-adversarial/docs/UNCERTAINTY.md)
* **Statistical Methods & Testing:** [`docs/STATISTICAL_METHODS.md`](docs/STATISTICAL_METHODS.md)
* **Threat Model:** [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)
