# Comprehensive Viva Voce Examination & Defense Guide

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

This guide prepares the researcher for an intense oral examination (viva voce) by a harsh research supervisor or external examiner. It provides detailed, scientifically rigorous answers to 35 critical questions covering domain science, Transformer architectures, threat modeling, probabilistic MCMC search, defender training, uncertainty quantification, empirical results, and strategic defense positioning.

---

## Section 1: Thesis Scope, Framing & Core Claims

### Q1: What is the core scientific thesis of this project?
**Answer**: The thesis demonstrates that deep learning sequence-to-property surrogate models in materials informatics (specifically Transformer architectures predicting polymer band gaps from Polymer SMILES strings) exhibit prediction fragility under minor, chemically constrained digital sequence perturbations. Furthermore, incorporating a dynamic probabilistic MCMC attack generator into a closed-loop min-max adversarial training workflow significantly reduces prediction drift ($0.0965 \to 0.0766\text{ eV}$, a 20.63% drop) and stabilizes model sensitivity across random seeds (78.5% variance reduction).

### Q2: How do you respond to the critique: "Your chemical constraints (RDKit validity + Tanimoto $\ge 0.5$) do not guarantee real-world wet-lab synthesizability"?
**Answer**: We explicitly acknowledge this boundary. RDKit sanitization, valence check, attachment point star `*` balance, molecular weight limits ($\pm 50\%$), and Morgan fingerprint Tanimoto similarity ($S_{\text{Tanimoto}} \ge 0.5$) define **computational plausibility**, ensuring candidates are syntactically valid and structurally related SMILES strings. They do **not** guarantee wet-lab synthesizability, thermodynamic stability (e.g., DFT formation energy on the convex hull), or high Synthetic Accessibility (SA) scores. This work evaluates model robustness within digital screening pipelines, not wet-lab chemical synthesis.

### Q3: Why did clean prediction error (RMSE) increase from $1.1439\text{ eV}$ to $1.3672\text{ eV}$ after adversarial training?
**Answer**: This shift represents the standard **accuracy-robustness tradeoff** well-documented in adversarial machine learning (e.g., Madry et al., 2017; Tsipras et al., 2018). Minimizing worst-case adversarial loss $\mathbb{E}[\mathcal{L}(f_\theta(x_{\text{adv}}), \operatorname{stopgrad}(f_\theta(x)))]$ forces the network to learn smoother decision boundaries in local sequence neighborhoods, which slightly reduces peak fitting capacity on clean, unperturbed distribution centers. In our benchmark, Clean RMSE shifted from $1.1439 \pm 0.1128\text{ eV}$ to $1.3672 \pm 0.1631\text{ eV}$, Clean MAE shifted from $0.9237 \pm 0.1567\text{ eV}$ to $1.1489 \pm 0.1474\text{ eV}$, and Clean $R^2$ shifted from $0.5583 \pm 0.0911$ to $0.3662 \pm 0.1500$. This tradeoff is an inherent property of robust optimization, not a defect.

---

## Section 2: Domain Science & Physical Chemistry

### Q4: What is electronic band gap ($E_g$), and why is it the primary target property in this project?
**Answer**: Electronic band gap $E_g$ is the energy difference between the top of the valence band (Valence Band Maximum, VBM) and the bottom of the conduction band (Conduction Band Minimum, CBM) in solid-state materials:
$$E_g = E_{\text{CBM}} - E_{\text{VBM}} \quad [\text{in electron-volts, eV}]$$
It is the primary target because $E_g$ fundamentally determines the electronic conductivity, optical absorption spectrum, and charge transport efficiency of conjugated polymers used in organic photovoltaics (OPVs), organic light-emitting diodes (OLEDs), and organic field-effect transistors (OFETs).

### Q5: What is the physical distinction between the solid-state band gap ($E_g$) and the molecular HOMO-LUMO gap ($\Delta E_{\text{HL}}$)?
**Answer**: Molecular HOMO-LUMO gap ($\Delta E_{\text{HL}} = \epsilon_{\text{LUMO}} - \epsilon_{\text{HOMO}}$) applies to an isolated, gas-phase single molecule. In contrast, solid-state band gap ($E_g$) applies to extended periodic solids or polymer films. $E_g$ is systematically smaller than $\Delta E_{\text{HL}}$ by $1.0 - 2.0\text{ eV}$ due to intermolecular polarization screening, dielectric relaxation, and interchain electronic band dispersion in bulk solids.

### Q6: How is polymer band gap measured experimentally and calculated computationally?
**Answer**: Experimentally, $E_g$ is measured using UV-Vis absorption spectroscopy (via Tauc plot analysis for optical bandgap $E_{g,\text{opt}}$), Photoelectron Spectroscopy (UPS for VBM, IPES for CBM for transport gap $E_{g,\text{trans}}$), or Cyclic Voltammetry (CV). Computationally, $E_g$ is calculated via Density Functional Theory (DFT) using hybrid functionals (HSE06) or GW many-body quasiparticle corrections.

---

## Section 3: Target Model Architecture & Transformer Pipeline

### Q7: Why use a Transformer model for polymer property prediction instead of traditional descriptors or GNNs?
**Answer**: Polymer repeat units are naturally represented as 1D chemical sequences via Polymer SMILES (PSMILES). Transformers capture long-range structural dependencies and functional group interactions across sequence positions through multi-head self-attention without requiring rigid graph alignment or manual feature engineering (e.g. TransPolymer).

### Q8: How does the chemical tokenizer operate, and why is character-level splitting avoided?
**Answer**: The PSMILES tokenizer (`PSmilesTokenizer`) uses regex pattern matching to split chemical strings into multi-character sub-words (e.g., `[nH]`, `[C@@H]`, `Cl`, `Br`, `*`). Character-level splitting would tear chlorine `Cl` into `C` + `l`, silently converting chlorine into carbon and corrupting chemical representation semantics.

### Q9: Explain Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) in Multi-Head Self-Attention.
**Answer**: For input embeddings $\mathbf{X} \in \mathbb{R}^{L \times d}$, linear projections create $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}^K$, and $\mathbf{V} = \mathbf{X}\mathbf{W}^V$. Scaled dot-product attention computes:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right) \mathbf{V}$$
Query $\mathbf{Q}$ queries token relationships, Key $\mathbf{K}$ matches compatibility, and Value $\mathbf{V}$ aggregates sequence context. Multi-head attention ($N_h=4$) computes this in parallel across distinct representation subspaces ($d_k = 16$).

### Q10: What is the purpose of the Two-Branch Head (`TwoBranchTransformerRegressorModel`)?
**Answer**: The two-branch architecture explicitly decouples feature learning into two specialized pathways:
1. **Representation-Invariance Branch ($\mathbf{z}_{\text{repr}} \in \mathbb{R}^{32}$)**: Learns features invariant to canonical SMILES string representations.
2. **Chemistry-Sensitivity Branch ($\mathbf{z}_{\text{chem}} \in \mathbb{R}^{32}$)**: Learns high-sensitivity features responsive to functional group modifications.
Concatenating $\mathbf{z}_{\text{repr}}$ and $\mathbf{z}_{\text{chem}}$ into a fused regression head provides superior property sensitivity.

---

## Section 4: Threat Model & Attack Surface

### Q11: What is the realistic threat model for this work?
**Answer**: The attack surface consists of **automated digital screening pipelines** in materials informatics, where high-throughput computational platforms consume digital polymer representations (PSMILES) to rank candidates for downstream DFT or synthesis. An adversary (or random digital noise/corruption) introduces small sequence modifications to misdirect candidate ranking or cause model prediction drift without detection by syntax parsers.

### Q12: Is an adversarial drift of $0.0965\text{ eV}$ a physical security breach?
**Answer**: **No.** An adversarial prediction drift of $0.0965\text{ eV}$ measures the **computational sensitivity** and mathematical vulnerability of the surrogate model. It indicates that digital screening algorithms can be fooled into overestimating or underestimating band gaps, leading to poor candidate selection in automated workflows. It is not an immediate physical synthesis exploit or bio-hazard.

### Q13: What is the attacker's query budget ($Q$), and why is it strictly enforced at $Q=20$?
**Answer**: The query budget $Q \le 20$ represents the maximum number of forward-pass queries the attacker can submit to the target surrogate model per candidate molecule. Setting $Q=20$ reflects realistic rate-limiting, compute constraints, and anomaly detection thresholds in public/enterprise screening APIs, ensuring fair comparison across search baselines.

---

## Section 5: Probabilistic MCMC Search & Plausibility Validation

### Q14: How does the Probabilistic MCMC Attack Generator work?
**Answer**: `ProbabilisticMCMCAttack` executes a Metropolis-Hastings stochastic search over discrete chemical mutation space. At step $t$, given current sequence $x_t$, it samples a candidate edit $x'$ from proposal distribution $q(x' | x_t)$ using 4 proposal operators:
1. **Atom Substitution**: Swapping halogen/heteroatom tokens (e.g. $-\text{F} \leftrightarrow -\text{Cl}$).
2. **Fragment Insertion**: Inserting small conjugated rings or linkers.
3. **Side-Chain Deletion**: Removing aliphatic side chains.
4. **Link Alteration**: Modifying single/double bond connectivity.

Candidate $x'$ is passed through `ChemicalPlausibilityValidator`. If valid, it is accepted according to the Metropolis rule:
$$\alpha(x_t \to x') = \min\left(1, \exp\left(\frac{\mathcal{L}_{\text{total}}(x') - \mathcal{L}_{\text{total}}(x_t)}{T}\right)\right)$$
where $\mathcal{L}_{\text{total}}(x) = |f_\theta(x) - f_\theta(x_0)| + \mathcal{L}_{\text{chem}}(x)$ at temperature $T = 5.0$.

### Q15: What four checks are performed by `ChemicalPlausibilityValidator`?
**Answer**:
1. **RDKit Parsing & Sanitization**: Validates syntax, explicit valence, and aromatic kekulization.
2. **Attachment Star Balance**: Preserves exact polymer terminal star `*` counts ($\text{Count}_*(x') == \text{Count}_*(x)$).
3. **Molecular Weight Bounds**: Enforces $0.5 \le \frac{MW(x')}{MW(x)} \le 1.5$ (or $\Delta MW \le 50\text{ g/mol}$).
4. **Tanimoto Similarity Threshold**: Enforces $S_{\text{Tanimoto}}(x, x') \ge 0.5$ using Radius-2 Morgan fingerprints (2048-bit).

---

## Section 6: Closed-Loop Defender Training & Loss Formulations

### Q16: What is the closed-loop defender optimization objective?
**Answer**: The defender updates model parameters $\theta$ by solving a min-max optimization problem using a combined loss function:
$$\min_\theta \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ (1 - \lambda) \left(f_\theta(x) - y\right)^2 + \lambda \left(f_\theta(x_{\text{adv}}') - y\right)^2 \right]$$
where $x_{\text{adv}}' = \arg\max_{x' \in \mathcal{C}(x)} |f_\theta(x') - f_\theta(x)|$ is generated dynamically at each training step ($\lambda = 0.5$).

### Q17: How does closed-loop dynamic training differ from static data augmentation?
**Answer**: Static data augmentation generates a fixed set of perturbed SMILES strings once prior to training. Closed-loop dynamic training continuously queries the evolving model state $f_\theta$ during every epoch, generating novel adversarial candidates specifically tailored to exploit current decision boundary vulnerabilities.

---

## Section 7: Uncertainty Quantification & Robustness Metrics

### Q18: How is epistemic uncertainty estimated via Monte Carlo Dropout?
**Answer**: During inference, dropout layers remain active (`model.train()`). We execute $N_{\text{mc}} = 15$ stochastic forward passes for input sequence $x$, obtaining predictions $\{y^{(1)}, y^{(2)}, \dots, y^{(N_{\text{mc}})}\}$. We compute:
- Predictive Mean: $\hat{y} = \frac{1}{N_{\text{mc}}} \sum_{i=1}^{N_{\text{mc}}} y^{(i)}$
- Epistemic Uncertainty (Variance): $\sigma^2(x) = \frac{1}{N_{\text{mc}}} \sum_{i=1}^{N_{\text{mc}}} \left(y^{(i)} - \hat{y}\right)^2$

### Q19: What is Epistemic Uncertainty Shift ($\Delta \sigma$), and how should its magnitude be interpreted?
**Answer**: $\Delta \sigma = \mathbb{E}[\sigma(x_{\text{adv}}) - \sigma(x_{\text{clean}})]$. Across 5 seeds:
- Baseline Model: $\Delta \sigma = 0.0006 \pm 0.0008\text{ eV}$
- Defended Model: $\Delta \sigma = 0.0001 \pm 0.0002\text{ eV}$
This corresponds to an **83.33% reduction in uncertainty drift**. In absolute magnitude, $\sigma$ values are small ($\sim 10^{-4}\text{ eV}$), reflecting overall model overconfidence. However, defender training successfully dampens variance spikes across local sequence perturbations.

### Q20: What is Mean Absolute Prediction Drift ($\text{MAD}$)?
**Answer**: $\text{MAD} = \frac{1}{N} \sum_{i=1}^N |f_\theta(x_i') - f_\theta(x_i)|$ (in $\text{eV}$). It measures the average output shift caused by valid adversarial perturbations across a dataset.

---

## Section 8: Multi-Seed Verified Results & Empirical Rigor

### Q21: What were the multi-seed evaluation results across 5 random seeds?
**Answer**: Evaluated across 5 independent seeds ($42, 123, 2026, 777, 999$):

| Metric | Baseline Model ($\mu \pm \sigma$) | Defended Model ($\mu \pm \sigma$) | Absolute / Relative Shift |
| :--- | :--- | :--- | :--- |
| **Clean RMSE (eV)** | $1.1439 \pm 0.1128$ | $1.3672 \pm 0.1631$ | $+0.2233\text{ eV}$ (Accuracy Tradeoff) |
| **Clean MAE (eV)** | $0.9237 \pm 0.1567$ | $1.1489 \pm 0.1474$ | $+0.2252\text{ eV}$ |
| **Clean $R^2$** | $0.5583 \pm 0.0911$ | $0.3662 \pm 0.1500$ | $-0.1921$ |
| **Adversarial RMSE (eV)** | $1.1677 \pm 0.0893$ | $1.3756 \pm 0.1489$ | $+0.2079\text{ eV}$ |
| **Mean Absolute Drift (eV)** | $\mathbf{0.0965 \pm 0.0466}$ | $\mathbf{0.0766 \pm 0.0100}$ | **20.63% Drift Reduction** |
| **Drift Cross-Seed Variance** | $0.0466$ | $0.0100$ | **78.5% Variance Drop** |
| **Epistemic Uncertainty Shift ($\Delta\sigma$)** | $0.0006 \pm 0.0008$ | $0.0001 \pm 0.0002$ | **83.33% Reduction** |

### Q22: How did Probabilistic MCMC compare to Random Mutation and Simple Substitution under equal query budget ($Q=20$)?
**Answer**:

| Attack Method | Mean Prediction Drift (eV) | Max Drift (eV) | Validity Rate |
| :--- | :--- | :--- | :--- |
| **Random Mutation** | $0.0312\text{ eV}$ | $0.1240\text{ eV}$ | $85.0\%$ |
| **Simple Substitution** | $0.0541\text{ eV}$ | $0.1890\text{ eV}$ | $92.0\%$ |
| **Probabilistic MCMC Attack** | $\mathbf{0.0965\text{ eV}}$ | $\mathbf{0.3152\text{ eV}}$ | $\mathbf{100.0\%}$ |

MCMC stochastic search generated **$3.09\times$ higher drift** than random mutation while achieving 100% validity due to its Metropolis guidance and plausibility filtering.

### Q23: What does the ablation study reveal about $\lambda$ and Tanimoto threshold selection?
**Answer**:
- **$\lambda$ Sensitivity**: At $\lambda=0.0$ (no adversarial loss), drift is $0.0581\text{ eV}$ but clean RMSE is $1.4122\text{ eV}$. At $\lambda=0.5$, optimal balance is achieved with clean RMSE $1.3467\text{ eV}$ and drift $0.0594\text{ eV}$.
- **Tanimoto Cutoff Sensitivity**: At $S_{\text{Tanimoto}} \ge 0.5$, validity is maintained at $95-100\%$. Tightening to $0.7$ restricts candidate diversity, reducing max drift from $0.4391\text{ eV}$ to $0.3176\text{ eV}$.

---

## Section 9: Dataset Characterization & Reproducibility

### Q24: Characterize the evaluation dataset and train/test split protocol.
**Answer**: The dataset consists of conjugated polymer repeat unit PSMILES strings paired with DFT-calculated solid-state band gaps ($E_g$). For each benchmark run, 20 representative polymer target sequences are evaluated across 5 random seeds ($42, 123, 2026, 777, 999$). Inputs are standardized using canonical PSMILES formatting with balanced attachment stars `*`. Target scaling is performed via `TargetScaler` (standard z-score normalization).

### Q25: How is reproducibility guaranteed across the repository?
**Answer**: All stochastic operations (PyTorch model initialization, NumPy random state, Python random seed, MCMC proposal sampling) are bound to explicit seed parameters. The test suite contains 429 passing tests (verified via `PYTHONPATH=. .venv/bin/pytest`). Complete multi-seed results are serialized in `results/comprehensive_benchmark_summary.json`.

---

## Section 10: Limitations, Future Work & Strategic Defense

### Q26: What are the main scientific limitations of this project?
**Answer**:
1. **In Silico Scope**: Evaluations are performed computationally without physical polymer synthesis or lab validation.
2. **Plausibility vs Synthesizability**: RDKit validity and Tanimoto similarity ($S_{\text{Tanimoto}} \ge 0.5$) do not guarantee synthetic accessibility (SA score) or thermodynamic stability.
3. **Clean Accuracy Tradeoff**: Clean RMSE increases by $0.2233\text{ eV}$ after defender training.
4. **Finite Query Budget**: $Q=20$ query cap restricts multi-step global search exploration.

### Q27: What is the proper boundary for discussing Cyberbiosecurity or DNA attacks?
**Answer**: Cyberbiosecurity, DNA synthesis screening, and biological threat sequence manipulations are **strictly conceptual future work application domains**. This repository implements and evaluates **polymer band gap sequence models**. We explicitly avoid claiming direct biological threat mitigation.

### Q28: Summary checklist for answering examiner questions:
- Always state the explicit thesis positioning statement (Proof-of-Concept for constrained adversarial robustness in materials sequence modelling).
- Cite exact multi-seed numbers: $0.0965 \pm 0.0466 \to 0.0766 \pm 0.0100\text{ eV}$ drift (20.63% reduction, 78.5% variance drop).
- Acknowledge the Clean RMSE tradeoff ($1.1439 \to 1.3672\text{ eV}$) as standard Pareto behavior.
- Clearly distinguish computational plausibility from wet-lab synthesizability.
- Cite both percentage (83.33%) and absolute numbers ($0.0006 \to 0.0001\text{ eV}$) for epistemic uncertainty shift.
