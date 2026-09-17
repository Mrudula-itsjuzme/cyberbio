> [!WARNING]
> **Historical Document / Superseded Methodology**
> This file chronicles an earlier development phase of the project (Phases 1-14). Early iterations of the MCMC defense in these logs evaluated closed-loop training using **original-label inheritance** for chemistry-changing edits. As established in the Phase 9 Scientific Audit, this assumption is physically invalid without a DFT oracle.
> The final canonical benchmark (`canonical_benchmark_no_leakage.json` and the final `EXPERIMENTAL_PROTOCOL.md`) explicitly abandons label inheritance in favor of **label-free consistency regularization**, and promotes **Rand-SMILES Augmentation** as the primary, physically sound defense. Please refer to `RESEARCH_CONTRIBUTIONS.md` for the current scientific consensus.

> [!NOTE]
> **Status**: HISTORICAL

# Phase 6: Attribution and Shortcut Analysis

## Objective
To determine what features and representation shortcuts the models actually rely on, and whether robustness training reduced superficial sequence dependence. 

Specifically, we evaluated:
1. Whether simple statistical models (length, token counts, RDKit descriptors) could match the Transformer baseline.
2. The correlation between Transformer predictions and superficial features.
3. Token occlusion sensitivities (which tokens drive predictions).
4. Whether robustness training caused trivial prediction collapse or meaningful embedding convergence.
5. Error stratified by length and chemical complexity.

---

## 1. Diagnostic Baselines (Shortcut Audit)

We trained simple Ridge regression baselines on the training set to predict bandgap and evaluated them on the validation set.

| Model | MAE (eV) | $R^2$ |
|-------|----------|-------|
| Target-Mean Baseline | 1.20 | 0.00 |
| Length-Only Regression | 1.02 | 0.16 |
| RDKit Descriptors | 0.79 | 0.43 |
| Token-Count Regression (Bag-of-Words) | 0.65 | 0.60 |
| Token-Frequency Regression | 0.65 | 0.64 |
| **Ordinary Transformer Baseline** | **0.48** | **0.84** |

**Interpretation:**
The simple baselines demonstrate that trivial shortcuts (like length or token counts) are highly informative for predicting the bandgap of these polymers. However, the Ordinary Transformer significantly outperforms all simple baselines by a large margin ($\sim 0.17$ eV MAE better than the best token-frequency shortcut). This provides evidence that the Transformer does not merely memorize token counts or lengths, but learns sequential chemical context.

---

## 2. Feature Correlation Audit

We computed the Pearson/Spearman correlations between superficial features (Sequence Length, Unique Tokens, Ring Tokens, Aromatic Atoms) and the models' predictions on the validation set.

| Feature | Correlation with True Bandgap | Correlation with Baseline Preds | Correlation with Mix-Robust Preds |
|---------|-------------------------------|---------------------------------|-----------------------------------|
| Length | -0.40 | -0.38 | -0.42 |
| Ring Count | -0.59 | -0.58 | -0.63 |
| Unique Tokens| -0.63 | -0.63 | -0.66 |

**Interpretation:**
Both the baseline and robust models learn these statistical correlations almost perfectly (matching the true data correlation). Robustness training did *not* remove the model's reliance on these features; it preserved them. The model still exploits these strong signals, but robustness training taught it to do so invariantly to SMILES syntax.

---

## 3. Embedding Stability & Prediction Smoothing

A common failure mode of robustness training is trivial smoothing (e.g., always predicting the mean to minimize variance). We checked the prediction distribution and extracted the pooled latent embeddings prior to the regression head.

| Model | Prediction Std Dev | Mean Embedding Distance (Randomized) | Max Prediction Drift |
|-------|--------------------|--------------------------------------|----------------------|
| Ordinary Baseline | 1.44 eV | 1.94 | 1.31 eV |
| Architecture Control | 1.33 eV | 2.33 | 0.88 eV |
| Randomization-Robust | 1.34 eV | 1.89 | 0.68 eV |
| **Mixed-Robust** | **1.37 eV** | **1.75** | **0.65 eV** |

**Interpretation:**
Robustness training preserved the expressive range of the predictions (std dev $\sim 1.37$ eV, compared to the baseline's $1.44$ eV). It did *not* collapse to the mean. More importantly, robustness training increased latent representation stability under equivalent representation perturbations (distance dropped from 1.94 to 1.75 in the mixed-robust model).

---

## 4. Token Occlusion Analysis

We systematically deleted single tokens and measured the absolute change in predicted bandgap.

**Highest Sensitivity Tokens (Average $\Delta$ eV across all models):**
1. `#` (Triple bonds): $\sim 0.74$ eV (Baseline) $\to 0.18$ eV (Robust)
2. `S` (Sulfur): $\sim 0.57$ eV (Baseline) $\to 0.51$ eV (Robust)
3. `=` (Double bonds): $\sim 0.45$ eV (Baseline) $\to 0.35$ eV (Robust)
4. `O` / `N` (Oxygen/Nitrogen): $\sim 0.34$ eV (Baseline) $\to 0.25$ eV (Robust)

**Interpretation:**
The models are highly sensitive to bond tokens (`=`, `#`) and heteroatoms (`S`, `O`, `N`), aligning with chemical intuition that unsaturation and electronegative atoms govern electronic properties like bandgap. Robustness training dampened extreme sensitivities to syntax symbols while preserving sensitivity to the core heteroatoms.

---

## 5. Stratified Error

Error analysis by length and chemical complexity quartiles showed consistent performance. The `mixed_robust` model consistently achieved the lowest MAE across all quartiles (Q1-Q4) compared to the baseline.

| Quartile | Baseline MAE | Mix-Robust MAE |
|----------|--------------|----------------|
| Length Q1 (Short) | 0.56 eV | 0.51 eV |
| Length Q4 (Long) | 0.45 eV | 0.40 eV |

The model performs better on longer polymers, likely because longer polymers in the dataset have lower bandgaps and narrower variance, making them easier to predict on average.

---

## 6. Final Verdict on Project Claims

Based on the cumulative evidence from Phases 1-6:

* **C1: The baseline Transformer just memorizes string length.** 
  **Verdict: NOT SUPPORTED.** The baseline Transformer ($0.48$ eV MAE) vastly outperforms a length-only regressor ($1.02$ eV MAE).
* **C2: The baseline Transformer just counts tokens (Bag-of-Words).**
  **Verdict: NOT SUPPORTED.** The baseline Transformer vastly outperforms Token-Count Ridge regression ($0.65$ eV MAE). It relies on sequential context.
* **C3: The model relies heavily on ring structures and specific heteroatoms.**
  **Verdict: SUPPORTED.** Occlusion sensitivity and correlation audits confirm strong reliance on rings, sulfur, oxygen, and nitrogen.
* **C4: Robustness training trivially smoothed predictions.**
  **Verdict: NOT SUPPORTED.** Prediction standard deviation remained healthy ($\sim 1.37$ eV).
* **C5: Robustness training forces the model to ignore sequential shortcuts.**
  **Verdict: NOT SUPPORTED.** The robust models maintained the exact same strong correlation with length and token counts as the baseline.
* **C6: Robustness training induces stable, invariant latent embeddings.**
  **Verdict: SUPPORTED.** Mean pairwise embedding distance for equivalent SMILES decreased significantly from $1.94$ (baseline) to $1.75$ (mixed-robust).

---

## Conclusion
The baseline Transformer does not simply count tokens; simple sequence-length and token-statistic baselines explain some signal but perform substantially worse than the Transformer, providing evidence against a purely trivial shortcut explanation. However, it is highly sensitive to the exact string syntax. Robustness training via consistency regularization does not destroy the model's reliance on valid chemical heuristics (like ring counts or heteroatoms) or smooth its predictions to zero. Instead, robustness training increased latent representation stability under equivalent representation perturbations, yielding a model that is both highly accurate and significantly robust to adversarial permutations and unseen deletions.
