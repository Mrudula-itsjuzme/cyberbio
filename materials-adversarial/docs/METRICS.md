# Evaluation Metrics and Robustness Equations

## 1. Regression Accuracy Metrics

### A. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Units**: Electron-volts ($\text{eV}$).
- **Desirable Direction**: Lower ($\downarrow$).
- **Interpretation**: Average absolute magnitude of bandgap prediction errors.

### B. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
- **Units**: Electron-volts ($\text{eV}$).
- **Desirable Direction**: Lower ($\downarrow$).
- **Interpretation**: Measures error magnitude while penalizing large outliers more severely.

### C. Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
- **Units**: Dimensionless (range $[-\infty, 1.0]$).
- **Desirable Direction**: Higher ($\uparrow$, max $1.0$).
- **Interpretation**: Proportion of target variance explained by model predictions.

---

## 2. Adversarial Robustness Metrics

### A. Mean Absolute Prediction Drift ($\text{MAD}$)
$$\text{MAD} = \frac{1}{N} \sum_{i=1}^N |f_\theta(x_i') - f_\theta(x_i)|$$
- **Units**: Electron-volts ($\text{eV}$).
- **Desirable Direction**: Lower ($\downarrow$).
- **Interpretation**: Primary robustness metric. Quantifies how much the model's bandgap prediction changes when input $x_i$ is perturbated into valid candidate $x_i'$.

### B. Maximum Absolute Prediction Drift ($\text{MaxDrift}$)
$$\text{MaxDrift} = \max_{i \in \{1,\dots,N\}} |f_\theta(x_i') - f_\theta(x_i)|$$
- **Units**: Electron-volts ($\text{eV}$).
- **Desirable Direction**: Lower ($\downarrow$).
- **Interpretation**: Worst-case prediction drift observed across all test samples.

### C. Attack Success Rate at Threshold $\tau$ ($\text{ASR}_\tau$)
$$\text{ASR}_\tau = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left[ |f_\theta(x_i') - f_\theta(x_i)| > \tau \right]$$
- **Units**: Fraction $[0.0, 1.0]$.
- **Desirable Direction**: Lower ($\downarrow$).
- **Default Thresholds**: $\tau = 0.5\text{ eV}$ and $\tau = 1.0\text{ eV}$.
- **Interpretation**: Percentage of attacks causing a bandgap prediction shift exceeding $\tau$.

### D. Candidate Validity Rate ($\text{VR}$)
$$\text{VR} = \frac{1}{N_{\text{generated}}} \sum_{j=1}^{N_{\text{generated}}} \mathbb{I}\left[ \text{ValidChemistry}(x_j') = \text{True} \right]$$
- **Units**: Fraction $[0.0, 1.0]$.
- **Desirable Direction**: Higher ($\uparrow$, ideal $1.0$).
- **Interpretation**: Proportion of generated candidates passing RDKit syntax, valence, attachment star `*` balance, and Tanimoto similarity filters ($S_{\text{Tanimoto}} \ge 0.5$).

### E. Prediction Drift Reduction Ratio ($\text{RR}_{\text{drift}}$)
$$\text{RR}_{\text{drift}} = \frac{\text{MAD}_{\text{baseline}} - \text{MAD}_{\text{defended}}}{\text{MAD}_{\text{baseline}}} \times 100\%$$
- **Units**: Percentage ($\%$).
- **Desirable Direction**: Higher ($\uparrow$).
- **Verified Value**: **20.63% reduction** across 5 seeds.

### F. Epistemic Uncertainty Shift ($\Delta \sigma$)
$$\Delta \sigma = \frac{1}{N} \sum_{i=1}^N \left( \sigma(x_i') - \sigma(x_i) \right)$$
- **Units**: Electron-volts ($\text{eV}$).
- **Desirable Direction**: Lower ($\downarrow$).
- **Verified Reduction**: **83.33% reduction** ($0.0006 \to 0.0001$).
