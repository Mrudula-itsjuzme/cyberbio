> [!WARNING]
> **Superseded by Final Benchmark**
> The statistics below reflect an earlier 5-seed, N=20 validation subset. The **canonical, leakage-free benchmark** results (which explicitly differentiate representation invariance from MCMC robustness regularization) are located in `results/experimental_summary.md`. Please refer to that document for the final citation-ready metrics.

# Comprehensive Verified Experimental Results

> [!IMPORTANT]
> **Scientific Integrity Notice**: Results in this section present ground-truth evaluation metrics for BOTH the full sealed test set ($N=632$) and the multi-seed benchmark subset ($N=20$).

---

## 1. Full Dataset Performance ($N=632$ Sealed Test Set)

When trained on the full 2,946 training set (`data/processed/splits.json`) with proper `TargetScaler` fitting on training data ONLY, the Transformer model (`TwoBranchTransformerRegressorModel`) achieves superior clean generalization:

| Metric | Full Test Set Value ($N=632$) | Physical Unit | Target Variance Explained ($R^2$) |
| :--- | :--- | :--- | :--- |
| **Clean Root Mean Squared Error (RMSE)** | **$0.6007\text{ eV}$** | $\text{eV}$ | — |
| **Clean Mean Absolute Error (MAE)** | **$0.4627\text{ eV}$** | $\text{eV}$ | — |
| **Clean Coefficient of Determination ($R^2$)** | **$0.8170$** | Dimensionless | **81.70% Variance Explained** |

---

## 2. Benchmark Subset Performance (5-Seed Validation, $N=20$)

Across 5 independent random seeds ($42, 123, 2026, 777, 999$) on the benchmark subset:

| Metric | Baseline Model ($\mu \pm \sigma$) | Defended Model ($\mu \pm \sigma$) | Absolute Change | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Clean RMSE (eV)** | $1.1439 \pm 0.1128$ | $1.3672 \pm 0.1631$ | $+0.2233\text{ eV}$ | Accuracy Tradeoff |
| **Clean MAE (eV)** | $0.9237 \pm 0.1567$ | $1.1489 \pm 0.1474$ | $+0.2252\text{ eV}$ | — |
| **Clean $R^2$** | $0.5583 \pm 0.0911$ | $0.3662 \pm 0.1500$ | $-0.1921$ | — |
| **Adversarial RMSE (eV)** | $1.1677 \pm 0.0893$ | $1.3756 \pm 0.1489$ | $+0.2079\text{ eV}$ | — |
| **Mean Absolute Drift (eV)** | $\mathbf{0.0965 \pm 0.0466}$ | $\mathbf{0.0766 \pm 0.0100}$ | $\mathbf{-0.0199\text{ eV}}$ | **20.63% Reduction** |
| **Drift Variance ($\sigma$)** | $0.0466$ | $0.0100$ | $-0.0366$ | **78.5% Variance Reduction** |
| **Epistemic Uncertainty Shift ($\Delta\sigma$)** | $0.0006 \pm 0.0008$ | $0.0001 \pm 0.0002$ | $-0.0005\text{ eV}$ | **83.33% Reduction** |

---

## 3. Expanded Attack Family Comparison (Query Budget $Q=20$)

Evaluating attack families against the baseline model:

| Attack Family | Search Method | Mean Drift (eV) | Max Drift (eV) | Candidate Validity Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Family A: Random Mutation** | Uniform token replacement | $0.0312\text{ eV}$ | $0.1240\text{ eV}$ | $85.0\%$ |
| **Family B: Deterministic Sub.** | Single bioisostere swaps | $0.0541\text{ eV}$ | $0.1890\text{ eV}$ | $92.0\%$ |
| **Family C: Probabilistic MCMC** | Metropolis-Hastings ($T=5$) | $\mathbf{0.0965\text{ eV}}$ | $\mathbf{0.3152\text{ eV}}$ | $\mathbf{100.0\%}$ |
| **Family D: SMILES Randomization**| Chemically identical SMILES | **$0.6147\text{ eV}$** | **$1.3622\text{ eV}$** | $\mathbf{100.0\%}$ |
| **Sequence-Length Baseline** | Linear regression on length | — | — | $R^2 = 0.2294$ |

### Insights:
- **SMILES Randomization Attack (Family D)** reveals massive representation vulnerability: querying chemically identical SMILES for the same polymer causes a **Mean Drift of 0.6147 eV**, proving the model relies heavily on SMILES syntax rather than pure 2D molecular graph invariants.
