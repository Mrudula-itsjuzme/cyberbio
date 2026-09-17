# Problem Statement & Research Objectives

## 1. Background and Motivation

Machine learning models, particularly sequence-based Transformers and Graph Neural Networks (GNNs), are increasingly deployed in materials informatics for rapid screening and property prediction of polymers, inorganic crystals, and organic molecules. These surrogates replace expensive density functional theory (DFT) computations or synthesis experiments.

However, deep learning surrogate models are known to be vulnerable to **adversarial perturbations**: minor, carefully chosen input modifications that cause disproportionately large prediction errors. In computer vision, adversarial perturbations manifest as high-frequency noise. In chemical sequence modelling (e.g., polymer SMILES/PSMILES), arbitrary character mutations are invalid because:
1. They frequently yield syntactically invalid SMILES strings that fail parser execution.
2. They generate chemically absurd or un-synthesizable molecular structures.
3. They violate fundamental chemical constraints such as valence rules and polymer connectivity.

Consequently, evaluating and enhancing the adversarial robustness of material property models requires a **unified framework** that couples **scientifically constrained adversarial attack generation** with **closed-loop defender training**.

---

## 2. Research Problem Definition

Given:
1. A dataset of polymer repeat unit representations $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, where $x_i \in \mathcal{X}$ is a PSMILES sequence string and $y_i \in \mathbb{R}$ is the ground-truth solid-state band gap $E_g$ in electron-volts ($\text{eV}$).
2. A sequence-to-property surrogate regression model $f_\theta: \mathcal{X} \to \mathbb{R}$ trained to minimize clean prediction error $\mathcal{L}_{\text{MSE}}(f_\theta(x), y)$.

The research problem addresses four fundamental vulnerabilities:

1. **Adversarial Fragility**: Standard clean-trained deep surrogate models exhibit high sensitivity to minor chemical mutations, producing severe prediction drift $|f_\theta(x') - f_\theta(x)| \gg 0$ on candidates $x'$ that remain chemically similar to $x$.
2. **Search Space Inefficiency of Naive Attacks**: Unconstrained random token mutations or simple deterministic substitutions produce either invalid SMILES or weak adversarial perturbations that fail to expose model blind spots under fixed query budgets $Q$.
3. **Uncertainty Blind Spots under Attack**: Surrogates often emit confident yet highly erroneous predictions on adversarial candidates without signaling elevated epistemic uncertainty.
4. **Lack of Defender Feedback**: Traditional evaluation treats attack generation and model training as disconnected systems, failing to provide defensive updates that systematically close adversarial vulnerabilities.

---

## 3. Core Research Objectives

To address these challenges, this project formulates and validates a unified adversarial attack and defender framework across six primary research goals:

1. **Formalize Constrained Adversarial Attacks**: Mathematically formulate the materials adversarial optimization objective balancing prediction error maximization against structural similarity ($S_{\text{Tanimoto}} \ge 0.5$) and chemical validity constraints.
2. **Develop Probabilistic MCMC Attack Generator**: Implement a discrete Markov Chain Monte Carlo (`ProbabilisticMCMCAttack`) generator using bioisosteric functional group proposal operators to efficiently explore valid adversarial candidate spaces.
3. **Establish Chemical Plausibility Validation**: Create a domain-aware plausibility validator (`ChemicalPlausibilityValidator`) enforcing RDKit parseability, valence correctness, polymer attachment star `*` balance, and molecular weight bounds.
4. **Implement Closed-Loop Adversarial Defender Training**: Formulate a min-max optimization loop where defender parameters $\theta$ are updated on dynamically generated adversarial candidates.
5. **Quantify Epistemic Uncertainty Shift**: Incorporate Monte Carlo Dropout (MC-Dropout) to measure predictive variance $\sigma^2(x)$ and evaluate uncertainty shift under attack ($\Delta \sigma$).
6. **Multi-Seed Statistical Verification**: Evaluate baseline vs defended models across 5 independent random seeds ($42, 123, 2026, 777, 999$) to demonstrate statistically significant robustness gains (mean $\pm$ std).

---

## 4. Scope and Explicit Non-Goals

### In-Scope Focus
- Solid-state polymer bandgap prediction ($E_g$ in eV).
- Sequence-to-property Transformer architectures (`TwoBranchTransformerRegressorModel`).
- Discrete chemical sequence mutations (substitutions, insertions, deletions, bioisosteric swaps).
- RDKit-based validity, Tanimoto similarity thresholding, and MC-Dropout uncertainty analysis.
- Closed-loop adversarial training and multi-seed statistical evaluation.

### Out-of-Scope (Future Work)
- Wet-lab experimental synthesis or physical sample fabrication.
- RAG/retrieval database poisoning attacks.
- Deepfake image generation or genomic sequence threat detection platforms.
- Multi-agent game-theoretic honeypots (designated as Phase 2 extensions).
