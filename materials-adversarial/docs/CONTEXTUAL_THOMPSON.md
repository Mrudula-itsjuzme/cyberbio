# Contextual Thompson Sampling

## 1. Motivation & Hypothesis
Based on prior observations that optimal adversarial edits are source-dependent, we hypothesized that simple, zero-shot structural features extracted from the source polymer can inform a Contextual Thompson controller, reducing early exploration costs (budget $\le 25$).

## 2. Features & Leakage Controls
- **Features Used**: Sequence length, ring count, branch count, aromatic fraction, heteroatom fraction, and normalized token frequencies (C, O, N, c, 1, 2, (, ), =).
- **Leakage Controls**: All features are extracted *before* any search operators execute. They depend strictly on the original SMILES string. Standard scaling is applied using statistics computed exclusively on the Train set. The cold-start prior is seeded identically and blindly for each unseen Test source.

## 3. Train/Validation/Test Split
A rigorous split of the dataset was produced using canonical boundaries, ensuring zero source overlap:
- **Train**: 150 sources (used to learn the context prior via independent fixed-arm sweeps).
- **Validation**: 30 sources.
- **Test**: 50 sources (held-out evaluation).
*(Saved in `results/contextual_thompson/context_source_split.json`)*

## 4. Model & Mathematics
We implemented a **Bayesian Linear Regression** prior.
For each arm $a$:
- Precision matrix: $\Lambda_a = \lambda I + \alpha X^T X$
- Mean vector: $\mu_a = \Lambda_a^{-1} (\alpha X^T y)$
- Sampling: $\hat{\theta}_a \sim \mathcal{N}(\mu_a, \Lambda_a^{-1})$
- Score: $S_a = \mathbf{z}(x)^T \hat{\theta}_a$

During online adaptation, $\Lambda_a$ and $\mu_a$ are updated via Sherman-Morrison rank-1 updates using observed rewards.

## 5. Evaluation Baselines
We compared Contextual Thompson against:
1. Uniform sampling
2. Non-Contextual Thompson (Beta-Bernoulli)
3. Global-Best Fixed Arm (the highest-yield arm across the train set)
4. Shuffled-Context Control (to isolate algorithmic artifacts from structural feature value)

## 6. Conclusion
*(Pending benchmark completion)*
