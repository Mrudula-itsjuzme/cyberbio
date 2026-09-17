> [!WARNING]
> **Historical Metrics / Superseded Baseline**
> The metrics and narratives in this document (e.g., 20.63% drift reduction, clean RMSE 1.1439, 83.33% uncertainty drop) belong to an earlier evaluation phase. 
> Please refer to `results/experimental_summary.md` and `results/canonical_benchmark_no_leakage.json` for the final, leakage-free canonical benchmark matrix, which explicitly separates representation-preserving augmentation from chemistry-changing MCMC stress tests.

# Ablation and Sensitivity Studies

## 1. Adversarial Loss Weight Sweep ($\lambda$)

The defender loss objective balances clean loss and adversarial loss via weight $\lambda \in [0.0, 1.0]$:
$$\mathcal{L} = (1 - \lambda) \mathcal{L}_{\text{clean}} + \lambda \mathcal{L}_{\text{adv}}$$

We evaluated model performance across five values of $\lambda$:

| Adversarial Weight ($\lambda$) | Clean RMSE (eV) | Adversarial RMSE (eV) | Mean Absolute Drift (eV) | Robustness Gain |
| :--- | :--- | :--- | :--- | :--- |
| **$\lambda = 0.00$ (Clean Only)** | $\mathbf{1.0995}$ | $1.1677$ | $0.0965$ | Baseline ($0.0\%$) |
| **$\lambda = 0.25$** | $1.2140$ | $1.2580$ | $0.0882$ | $8.60\%$ Drift Drop |
| **$\lambda = 0.50$ (Default)** | $1.3461$ | $1.3032$ | $\mathbf{0.0766}$ | **20.63% Drift Drop** |
| **$\lambda = 0.75$** | $1.4820$ | $1.4120$ | $0.0754$ | $21.86\%$ Drift Drop |
| **$\lambda = 1.00$ (Pure Adv)** | $1.6950$ | $1.5840$ | $0.0741$ | $23.21\%$ (Severe Clean Degradation) |

### Key Observations:
- **Optimal Balance at $\lambda = 0.50$**: Provides significant robustness gains (**20.63% drift reduction**) while avoiding the severe clean accuracy degradation seen at $\lambda = 1.00$.
- **Diminishing Returns**: Increasing $\lambda$ from $0.50$ to $1.00$ yields marginal drift improvements ($0.0766\text{ eV} \to 0.0741\text{ eV}$) at the cost of a large jump in clean RMSE ($1.3461\text{ eV} \to 1.6950\text{ eV}$).

---

## 2. Tanimoto Similarity Threshold Sweep ($S_{\text{Tanimoto}}$)

The similarity constraint enforces structural proximity ($S_{\text{Tanimoto}}(x, x') \ge S_{\text{min}}$). We evaluated four threshold levels:

| Threshold ($S_{\text{min}}$) | Mean Prediction Drift (eV) | Candidate Validity Rate | Scientific Plausibility |
| :--- | :--- | :--- | :--- |
| **$S_{\text{min}} = 0.0$ (Unconstrained)** | $0.1852\text{ eV}$ | $0.7200$ | Poor (Off-target chemical class shifts) |
| **$S_{\text{min}} = 0.3$ (Weak)** | $0.1340\text{ eV}$ | $0.8800$ | Moderate |
| **$S_{\text{min}} = 0.5$ (Default)** | $\mathbf{0.0965\text{ eV}}$ | $\mathbf{1.0000}$ | **High (Plausible functional group swaps)** |
| **$S_{\text{min}} = 0.7$ (Strict)** | $0.0410\text{ eV}$ | $1.0000$ | Very High (Minor atomic edits only) |

### Key Observations:
- **$S_{\text{min}} = 0.5$ Is Optimal**: Balances search freedom (allowing functional group bioisosteric substitutions) while guaranteeing 100% valid candidate structures.
- **$S_{\text{min}} = 0.0$ Causes Off-Target Shift**: Without a similarity cutoff, candidates drift into unrelated chemical classes, invalidating the adversarial assumption.

---

## 3. MCMC Attack Step Budget Sensitivity ($N_{\text{steps}}$)

We evaluated prediction drift on the baseline model across MCMC step budgets $N_{\text{steps}} \in \{5, 10, 20, 50, 100\}$:

| MCMC Steps ($N_{\text{steps}}$) | Mean Absolute Drift (eV) | Max Absolute Drift (eV) | Execution Time / Sample |
| :--- | :--- | :--- | :--- |
| **$N_{\text{steps}} = 5$** | $0.0421\text{ eV}$ | $0.1310\text{ eV}$ | $0.12\text{ s}$ |
| **$N_{\text{steps}} = 10$** | $0.0685\text{ eV}$ | $0.2140\text{ eV}$ | $0.24\text{ s}$ |
| **$N_{\text{steps}} = 20$ (Default)** | $\mathbf{0.0965\text{ eV}}$ | $\mathbf{0.3152\text{ eV}}$ | $0.48\text{ s}$ |
| **$N_{\text{steps}} = 50$** | $0.1120\text{ eV}$ | $0.3480\text{ eV}$ | $1.20\text{ s}$ |
| **$N_{\text{steps}} = 100$** | $0.1185\text{ eV}$ | $0.3590\text{ eV}$ | $2.42\text{ s}$ |

### Key Observations:
- Prediction drift increases rapidly from $N_{\text{steps}} = 5$ to $N_{\text{steps}} = 20$, then reaches a plateau above 50 steps.
- $N_{\text{steps}} = 20$ provides an ideal tradeoff between attack strength and computational efficiency.
