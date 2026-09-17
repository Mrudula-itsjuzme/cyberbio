# Structured Academic Literature Review

## 1. Literature Survey Overview

This literature review synthesizes key advancements at the intersection of materials informatics, Transformer language models, out-of-distribution (OOD) generalization, adversarial machine learning, and uncertainty quantification. 

Six foundational papers establish the theoretical grounding and technical baseline for this project:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOUNDATIONAL LITERATURE                                  │
├───────────────────────────────────────────┬─────────────────────────────────────────────┤
│ 1. Materials Informatics Methods (2023)  │ 2. TransPolymer Sequence LM (2023)          │
│    InfoMat / Wiley                        │    npj Computational Materials              │
│    (Surrogate modeling & pipeline gaps)   │    (PSMILES sequence Transformer baseline)  │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ 3. OOD Property Prediction (2025)         │ 4. Graph Virtual Adversarial Train (2025)   │
│    J. Phys. Chem. C                       │    Alexandria Engineering Journal           │
│    (Adversarial OOD exposure methods)     │    (Semi-supervised GNN virtual attacks)    │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ 5. Cross-Adversarial Generation (2022)    │ 6. Uncertainty Quantification (2024)        │
│    Frontiers in Pharmacology              │    Scientific Reports                       │
│    (Adversarial molecular generation)     │    (Bayesian UQ & variance estimation)      │
└───────────────────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 2. In-Depth Analysis of Core Literature

### Paper 1: Methods, Progresses, and Opportunities of Materials Informatics
- **Citation**: InfoMat / Wiley, 2023.
- **Problem Solved**: Reviews machine learning pipelines for predicting electronic, mechanical, and thermal properties of functional materials, detailing descriptor representation choices (fingerprints, graphs, SMILES).
- **Methodology**: Evaluates supervised learning, feature engineering, and high-throughput screening across inorganic and organic datasets.
- **Remaining Limitation**: Focuses primarily on clean in-distribution performance and standard benchmark accuracy without considering model vulnerability to intentional adversarial perturbations or malicious inputs.
- **Project Motivation**: Highlights the critical need to evaluate machine learning surrogates under adversarial stress conditions rather than relying solely on clean cross-validation metrics.

### Paper 2: TransPolymer: A Transformer-Based Language Model for Polymer Property Predictions
- **Citation**: *npj Computational Materials*, 2023.
- **Problem Solved**: Demonstrates that 1D PSMILES sequence representations processed through Transformer encoder architectures achieve state-of-the-art accuracy across 26 polymer properties.
- **Methodology**: Pre-trains a sequence Transformer on polymer SMILES strings and fine-tunes regression heads for target properties such as band gap ($E_g$) and glass transition temperature ($T_g$).
- **Remaining Limitation**: Assumes test inputs are drawn from identical chemical distributions and does not incorporate defensive adversarial training or chemical plausibility constraints during model optimization.
- **Project Motivation**: Directly inspires the target model architecture (`TwoBranchTransformerRegressorModel`), serving as the canonical sequence Transformer baseline for our adversarial attack and defense framework.

### Paper 3: Out-of-Distribution Material Property Prediction Using Adversarial Learning
- **Citation**: *Journal of Physical Chemistry C*, 2025.
- **Problem Solved**: Addresses performance degradation when material property predictors encounter out-of-distribution (OOD) chemical structures.
- **Methodology**: Employs domain-adversarial neural networks (DANN) to align feature representations between source and target material domains.
- **Remaining Limitation**: Focuses on passive domain shift rather than active, targeted adversarial attacks that attempt to maximize prediction error under strict chemical validity constraints.
- **Project Motivation**: Establishes that adversarial loss objectives can improve model generalization, motivating our closed-loop min-max defender training formulation.

### Paper 4: Semi-Supervised Learning-Based Virtual Adversarial Training on Graph for Molecular Property Prediction
- **Citation**: *Alexandria Engineering Journal*, 2025.
- **Problem Solved**: Improves GNN property prediction robustness using Virtual Adversarial Training (VAT) on molecular graphs.
- **Methodology**: Generates smooth continuous perturbations in feature embedding spaces to regularize model predictions on unlabelled molecules.
- **Remaining Limitation**: Continuous gradient perturbations in latent space do not map directly to valid discrete SMILES strings or real molecular structures, creating a disconnect between feature perturbations and real chemistry.
- **Project Motivation**: Motivates our discrete, sequence-level attack generator (`ProbabilisticMCMCAttack`) that operates directly on valid PSMILES strings using RDKit chemical validation.

### Paper 5: Cross-Adversarial Learning for Molecular Generation in Drug Design
- **Citation**: *Frontiers in Pharmacology*, 2022.
- **Problem Solved**: Utilizes adversarial generator-discriminator pairs to explore chemical space and generate bioactive candidate molecules.
- **Methodology**: Combines reinforcement learning rewards with adversarial loss functions to propose novel molecular structures.
- **Remaining Limitation**: Designed for generative drug discovery rather than evaluating property model fragility or training robust regression defenders.
- **Project Motivation**: Demonstrates that adversarial search can successfully explore chemical candidate space, inspiring our Metropolis-Hastings proposal mechanism.

### Paper 6: Uncertainty Quantification in Multivariable Regression for Material Property Prediction with Bayesian Neural Networks
- **Citation**: *Scientific Reports*, 2024.
- **Problem Solved**: Formulates uncertainty quantification (UQ) to identify when deep learning property models emit unreliable predictions.
- **Methodology**: Implements Bayesian Neural Networks (BNNs) and Monte Carlo Dropout to estimate predictive variance.
- **Remaining Limitation**: Evaluates UQ only on clean test data and passive OOD samples, without measuring how predictive uncertainty behaves under active adversarial attacks.
- **Project Motivation**: Guides our implementation of MC-Dropout uncertainty quantification to measure epistemic variance shift ($\Delta \sigma$) under adversarial attack.

---

## 3. Comparative Literature Summary

| Study | Target Representation | Attack / Perturbation Type | Chemical Validity Enforced? | Defender Training Included? | UQ Evaluated? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **InfoMat (2023)** | Fingerprints / Graphs | None (Survey) | N/A | No | No |
| **TransPolymer (2023)** | PSMILES Sequence | None (Clean Model) | N/A | No | No |
| **J. Phys. Chem. C (2025)**| Composition Vectors | Domain Shift (DANN) | No | Passive Domain Align | No |
| **Alex. Eng. J. (2025)** | Molecular Graph | Continuous Latent VAT | No (Latent only) | Yes (Virtual) | No |
| **Front. Pharmacol. (2022)**| SMILES / Graph | Generative GAN | Partially | No (Generator focus)| No |
| **Sci. Rep. (2024)** | Tabular / Descriptors | None (Clean UQ) | N/A | No | Yes (Clean) |
| **Our Unified Framework** | **PSMILES Sequence** | **Probabilistic MCMC Sequence** | **Yes (RDKit + Tanimoto $\ge 0.5$)** | **Yes (Closed-Loop Min-Max)** | **Yes (MC-Dropout $\Delta\sigma$)** |

---

## 4. Synthesis of Research Gap

The literature reveals a clear **three-fold research gap**:
1. **Disconnect Between Latent Perturbations and Real Chemistry**: Prior adversarial methods modify continuous latent embeddings rather than generating discrete, syntactically and chemically valid material sequence candidates ($x' \in \mathcal{X}$).
2. **Absence of Unified Attack-Defender Frameworks in Materials**: Existing studies treat property prediction, attack generation, and defensive training as isolated tasks rather than a closed-loop min-max system.
3. **Lack of Uncertainty Analysis Under Attack**: Existing uncertainty quantification methods evaluate variance only on clean datasets, ignoring model overconfidence under adversarial stress.

This project bridges these gaps by constructing a **unified adversarial learning framework** that combines probabilistic MCMC attack generation, strict chemical plausibility filtering, closed-loop min-max defender training, and MC-Dropout uncertainty quantification.
