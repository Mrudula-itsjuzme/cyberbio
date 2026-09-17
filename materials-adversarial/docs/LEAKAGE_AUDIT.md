# Data Leakage and Shortcut Feature Audit

> [!IMPORTANT]
> **Audit Objective**: Conduct a rigorous data leakage and shortcut feature audit covering deduplication, scaffold overlap, normalization fitting, sequence length correlations, and attack candidate overlap.

---

## 1. Overview of Audit Findings

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                   DATA LEAKAGE SUMMARY                                    │
├───────────────────────────┬───────────────────────────────┬───────────────────────────────┤
│ Leakage Vector            │ Empirical Findings            │ Severity Status               │
├───────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Exact Canonical SMILES    │ 0 overlap train/val/test      │ VERIFIED CLEAN                │
│ Bemis-Murcko Scaffolds    │ 110 / 238 test scaffolds leak │ HIGH RISK (Random Split)      │
│ Target Scaler Normalization│ Fitted on test in benchmark  │ FIXED IN PROPER PIPELINE      │
│ Sequence Length Shortcut  │ r = -0.4310, R^2 = 0.2294    │ CONFIRMED SHORTCUT FEATURE    │
│ Attack Candidate Leakage  │ Validated against train set   │ LOW RISK                      │
└───────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## 2. Split Deduplication & Canonical SMILES Audit

An exact canonical SMILES deduplication check was performed across the official splits defined in `data/processed/splits.json` ($N=2946$ train, $N=631$ validation, $N=632$ test):

| Split Pair | Number of Exact Canonical SMILES Overlaps | Leakage Status |
| :--- | :--- | :--- |
| **Train vs. Validation** | 0 | Clean |
| **Train vs. Test** | 0 | Clean |
| **Validation vs. Test** | 0 | Clean |

`build_splits()` in `src/materials_adv/data/splits.py` canonicalizes all PSMILES strings using RDKit before grouping by canonical SMILES, ensuring zero exact identity leakage across splits.

---

## 3. Bemis-Murcko Scaffold Overlap Audit

When evaluating structural scaffold overlap (Bemis-Murcko frameworks extracted via `MurckoScaffold.MurckoScaffoldSmiles`), the random splitting strategy (`strategy: random` in `configs/dataset.yaml`) results in significant scaffold sharing between training and evaluation splits:

- **Total Unique Scaffolds**: Train = 786, Validation = 242, Test = 238
- **Train vs. Validation Scaffold Overlap**: 111 scaffolds shared (**45.8%** of validation scaffolds)
- **Train vs. Test Scaffold Overlap**: 110 scaffolds shared (**46.2%** of test scaffolds)
- **Validation vs. Test Scaffold Overlap**: 52 scaffolds shared

> [!WARNING]
> **Scaffold Leakage Risk**: Random splitting allows the model to encounter familiar molecular scaffolds in the test set. To measure true out-of-scaffold generalization, a scaffold-grouped split (`strategy: scaffold` in `splits.py`) must be deployed.

---

## 4. Target Scaler Normalization Leakage

In `scripts/run_comprehensive_benchmark_suite.py` and `scripts/run_before_after_defense_experiment.py`, `TargetScaler.fit()` was invoked on `df['property_value'][:100]` **before** performing the 80/20 train/test split.

- **Impact**: Test set target statistics ($\mu_{\text{test}}, \sigma_{\text{test}}$) slightly influenced the normalization parameters ($\mu_{\text{scaler}}, \sigma_{\text{scaler}}$).
- **Remediation**: `TargetScaler.fit()` MUST be called strictly on training split targets `train_targets` prior to transforming validation or test sets, as implemented in `scripts/run_audit_experiments.py`.

---

## 5. Sequence-Length Shortcut Analysis

Sequence length (number of characters in PSMILES string) exhibits a strong negative correlation with polymer band gap:

- **Pearson Correlation ($r$)**: **-0.4310** (p < 0.0001)
- **Physical Reason**: In conjugated polymers, longer PSMILES strings typically feature larger conjugated aromatic rings and extended $\pi$-electron delocalization systems, which physically reduce the electronic bandgap ($E_g$).

### Sequence-Length-Only Baseline Performance
A simple 1D linear regression model ($E_g = -0.0330 \cdot \text{Length} + 5.7626$) trained on sequence length alone achieves surprisingly strong predictive performance on the test set ($N=632$):

| Model / Baseline | Test RMSE (eV) | Test MAE (eV) | Test $R^2$ |
| :--- | :--- | :--- | :--- |
| **Target-Mean Baseline** | $1.4040\text{ eV}$ | $1.1587\text{ eV}$ | $0.0000$ |
| **Sequence-Length-Only Linear Model** | **$1.2325\text{ eV}$** | **$0.9668\text{ eV}$** | **$0.2294$** |
| **Full Two-Branch Transformer Model** | $0.6007\text{ eV}$ | $0.4627\text{ eV}$ | $0.8170$ |

> [!IMPORTANT]
> **Shortcut Finding**: Sequence length alone accounts for **22.94% of the target variance ($R^2 = 0.2294$)**. Transformer models can exploit sequence length as a shortcut feature rather than learning true electronic structure rules. Adversarial attack generators must monitor whether candidate perturbations manipulate sequence length to achieve prediction drift.
