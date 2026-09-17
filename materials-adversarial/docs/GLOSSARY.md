# Glossary of Symbols and Technical Notation

## 1. Physical & Chemical Terminology

- **$E_g$**: Solid-state electronic band gap in electron-volts ($\text{eV}$), defined as $E_g = E_{\text{CBM}} - E_{\text{VBM}}$.
- **$\Delta E_{\text{HL}}$**: Molecular HOMO-LUMO gap in electron-volts ($\text{eV}$), defined as $\Delta E_{\text{HL}} = \epsilon_{\text{LUMO}} - \epsilon_{\text{HOMO}}$.
- **$E_{\text{CBM}}$**: Energy level of the Conduction Band Minimum.
- **$E_{\text{VBM}}$**: Energy level of the Valence Band Maximum.
- **PSMILES**: Polymer Simplified Molecular-Input Line-Entry System string, representing polymer repeat units with wildcard attachment stars `*`.
- **$S_{\text{Tanimoto}}$**: Tanimoto structural similarity coefficient computed on Morgan circular fingerprints (range $[0.0, 1.0]$).
- **Bioisostere**: Chemical substituent or group with similar physical/chemical properties producing broadly similar biological or electronic effects (e.g. $-\text{F} \leftrightarrow -\text{Cl}$).

---

## 2. Machine Learning & Transformer Notation

- **$x$**: Clean polymer sequence representation string.
- **$x'$ / $x_{\text{adv}}$**: Adversarial polymer sequence candidate string.
- **$y$**: Ground-truth target band gap value in $\text{eV}$.
- **$\hat{y}$ / $f_\theta(x)$**: Model prediction for sequence $x$ in $\text{eV}$.
- **$\theta$**: Trainable parameters of the neural network surrogate.
- **$d_{\text{model}}$**: Transformer hidden embedding dimension ($64$).
- **$N_h$**: Number of attention heads in multi-head self-attention ($4$).
- **$N_{\text{layers}}$**: Number of stacked Transformer encoder layers ($2$).
- **$\mathbf{z}_{\text{repr}}$**: Representation-invariance branch embedding vector ($\mathbb{R}^{32}$).
- **$\mathbf{z}_{\text{chem}}$**: Chemistry-sensitivity branch embedding vector ($\mathbb{R}^{32}$).

---

## 3. Adversarial & Uncertainty Notation

- **$Q$**: Maximum allowable query budget per source molecule ($Q \le 20$).
- **$N_{\text{steps}}$**: Number of search steps in Metropolis MCMC candidate generation ($20$).
- **$T$**: Search temperature in Metropolis acceptance rule ($5.0$).
- **$\lambda$**: Adversarial loss weight parameter in defender training ($\lambda \in [0.0, 1.0]$, default $0.5$).
- **$\text{MAD}$**: Mean Absolute Prediction Drift ($\mathbb{E}[|f_\theta(x') - f_\theta(x)|]$).
- **$\text{ASR}_\tau$**: Attack Success Rate at drift threshold $\tau$ (fraction of attacks exceeding $\tau$).
- **$N_{\text{mc}}$**: Number of stochastic Monte Carlo Dropout forward passes ($15$).
- **$\sigma^2(x)$**: Epistemic prediction variance estimated via MC-Dropout.
- **$\Delta \sigma$**: Epistemic uncertainty shift under adversarial attack ($\mathbb{E}[\sigma(x_{\text{adv}}) - \sigma(x_{\text{clean}})]$).
