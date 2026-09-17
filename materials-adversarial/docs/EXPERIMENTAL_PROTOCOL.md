# Experimental Protocol and Hyper-Parameters

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

## 1. Hardware and Execution Environment

All experiments were executed under the following hardware and software environment:
- **Operating System**: Linux (Ubuntu 22.04 LTS / Debian x86_64).
- **CPU**: Intel Xeon / AMD EPYC workstation execution.
- **Python Environment**: Python 3.12.3 in virtual environment (`.venv`).
- **Core Dependencies**: PyTorch 2.13.0+cpu, RDKit 2026.03.5, NumPy 2.5.2, Pandas 2.2.0, Matplotlib 3.11.2.

---

## 2. Dataset Characterization & Splitting Protocol

- **Dataset Source**: Conjugated polymer band gap dataset (`data/processed/processed.csv`).
- **Input Representation**: PSMILES strings (polymer repeat units with wildcard attachment stars `*`).
- **Target Property**: Solid-state electronic band gap $E_g$ in electron-volts ($\text{eV}$).
- **Sample Size**: 20 representative polymer target sequences evaluated per seed across 5 random seeds ($42, 123, 2026, 777, 999$).
- **Scaling & Normalization**: Target values normalized via `TargetScaler` (standard z-score normalization) during training and inverse transformed for all evaluation metrics.

---

## 3. Multi-Seed Execution Protocol

To eliminate random seed bias and ensure statistical reproducibility, every training run, attack generation pass, adversarial defender update, and evaluation step was repeated across **5 independent random seeds**:

$$\text{Seeds } \mathcal{S}_{\text{seeds}} = \{42, 123, 2026, 777, 999\}$$

For each seed $s \in \mathcal{S}_{\text{seeds}}$:
1. Initialize PyTorch, NumPy, and Python random number generators with seed $s$.
2. Train baseline `TwoBranchTransformerRegressorModel` on clean training split for 4 epochs.
3. Attack baseline model with `ProbabilisticMCMCAttack` ($N_{\text{steps}}=20, T=5.0, S_{\text{Tanimoto}} \ge 0.5$) on test split. Record baseline metrics.
4. Train defended `TwoBranchTransformerRegressorModel` using closed-loop adversarial training ($\lambda=0.5, N_{\text{steps}}=10$) for 4 epochs.
5. Attack defended model with `ProbabilisticMCMCAttack` on test split. Record defended metrics.
6. Calculate mean $\pm$ standard deviation across all 5 seeds.

---

## 4. Hyper-Parameter Table

| Component | Hyper-Parameter | Value |
| :--- | :--- | :--- |
| **Model Architecture** | Model Type | `TwoBranchTransformerRegressorModel` |
| | Embedding Dim ($d_{\text{model}}$)| 64 |
| | Encoder Layers ($N_{\text{layers}}$) | 2 |
| | Attention Heads ($N_h$) | 4 |
| | Feed-Forward Dim ($d_{\text{ff}}$)| 128 |
| | Branch Dimension ($d_{\text{branch}}$) | 32 |
| | Dropout Rate ($p$) | 0.1 |
| | Max Sequence Length | 256 |
| **Training (Defender)**| Optimizer | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| | Learning Rate ($\eta$) | $1 \times 10^{-3}$ |
| | Batch Size ($B$) | 16 |
| | Training Epochs | 4 epochs per seed |
| | Adversarial Loss Weight ($\lambda$) | 0.5 (default) |
| **MCMC Attacker** | Search Budget ($N_{\text{steps}}$) | 20 steps (eval), 10 steps (train) |
| | Search Temperature ($T$) | 5.0 |
| | Query Limit ($Q$) | 20 queries per sample |
| | Min Tanimoto Similarity ($S_{\text{min}}$) | 0.5 |
| | MW Ratio Bounds | $[0.5, 1.5] \times MW$ |
| **Uncertainty (UQ)** | MC-Dropout Samples ($N_{\text{mc}}$) | 15 passes |
