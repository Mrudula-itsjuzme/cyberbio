# Scientific Consistency and Task Provenance Audit

> [!IMPORTANT]
> **Audit Objective**: Establish the ground truth of the dataset, target property, physical units, model task, and experimental artifacts. Resolve all contradictions across historical project documentation.

---

## 1. Ground Truth Prediction Task Identity

Through direct code and dataset inspection (`data/raw/bandgap_chain.csv`, `data/processed/processed.csv`, `configs/dataset.yaml`, `src/materials_adv/data/preprocessing.py`), the ground truth prediction task is established as follows:

| Property Attribute | Ground Truth Definition | Empirical Evidence / Source File |
| :--- | :--- | :--- |
| **Material System** | Conjugated polymer repeat units | `data/processed/processed.csv` (`original_representation`) |
| **Input Representation** | 1D Polymer SMILES (PSMILES) strings with terminal stars `*` | `src/materials_adv/data/tokenizer.py` (`PSmilesTokenizer`) |
| **Target Property** | Chain Electronic Band Gap ($E_g$) | `configs/dataset.yaml` (`target_property: "Bandgap"`) |
| **Physical Unit** | Electron-volts ($\text{eV}$) | `data/processed/processed.csv` (`units: "eV"`) |
| **Source Dataset** | polyVERSE polymer bandgap dataset (Ramprasad Group) | `data/raw/bandgap_chain.csv` ($N=4209$) |
| **Target Value Range** | Min: $0.0710\text{ eV}$, Max: $9.8351\text{ eV}$ | Computed directly from `data/processed/processed.csv` |
| **Target Distribution** | Mean: $4.4861\text{ eV}$, Std: $1.4460\text{ eV}$, Median: $4.5756\text{ eV}$ | Computed directly from `data/processed/processed.csv` |
| **Numerical Meaning** | Predicted chain electronic bandgap ($E_g$) of isolated repeat unit | `configs/dataset.yaml` & `data/processed/processed.csv` |

---

## 2. Scientific Claim & Provenance Audit Table

| Claim / Concept | Source in Code / Data | Historical / Current Docs | Consistent? | Required Fix / Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| **Target Property: Band Gap ($E_g$ in eV)** | `configs/dataset.yaml` (`bandgap_chain`), `data/processed/processed.csv` | `docs/PROJECT_OVERVIEW.md`, `docs/RESEARCH_PAPER_DRAFT.md` | **Consistent** | Maintain $E_g$ (eV) across all active documentation. |
| **Glass Transition Temp ($T_g$ in K)** | Present ONLY in literature secondary file `final_polymer_properties_fromliterature.csv` ($N=443$) | Legacy code comments in `preprocessing.py` and old notes | **Inconsistent** | Explicitly note that $T_g$ was explored in early pilot phases but is NOT the target of the canonical model. |
| **Dataset Size ($N=4209$)** | `data/processed/processed.csv` ($N=4209$) | `docs/EXPERIMENTAL_PROTOCOL.md` cited $N=100$ truncated benchmark | **Inconsistent** | Clarify that full dataset is $N=4209$ ($2946$ train / $631$ val / $632$ test). Truncated benchmark script (`reps[:100]`) evaluated $N=20$ test slice. |
| **Full Test Set Clean Performance** | Model trained on $N=2946$ full split achieves Clean RMSE = $0.6007\text{ eV}$, MAE = $0.4627\text{ eV}$, $R^2 = 0.8170$ | Earlier docs reported truncated $N=20$ benchmark (Clean RMSE $1.1439\text{ eV}$) | **Inconsistent** | Report BOTH full test set ($N=632$) performance and truncated benchmark subset ($N=20$) performance explicitly. |
| **Target Normalization Scaling** | `TargetScaler` in `src/materials_adv/data/scaler.py` | `docs/METRICS.md`, `docs/RESULTS.md` | **Consistent** | All metrics are reported in unscaled physical units ($\text{eV}$) after inverse z-score transformation. |
| **DFT / Quasiparticle $E_g$ Calculations** | Dataset targets derived from polyVERSE chain bandgap computations | `docs/MATERIALS_SCIENCE_BACKGROUND.md`, `docs/VIVA_GUIDE.md` | **Consistent** | Clarify that labels are computational chain bandgaps from polyVERSE, not solid-state bulk measurements. |

---

## 3. Discrepancy Analysis & Reconciliation

1. **Glass Transition Temperature ($T_g$) vs Band Gap ($E_g$)**:
   - In early exploratory commits, `final_polymer_properties_fromliterature.csv` was analyzed for multi-property screening (including $T_g$ in Kelvin). However, the primary project repository locked onto `bandgap_chain.csv` ($N=4209$), which measures polymer chain bandgap ($E_g$) in $\text{eV}$.
2. **Benchmark Subset ($N=20$) vs Sealed Test Set ($N=632$)**:
   - `scripts/run_comprehensive_benchmark_suite.py` truncated data to the first 100 rows (`reps[:100]`) for rapid multi-seed benchmark execution (80 train / 20 test).
   - Training on the full 2,946 train set yields **Clean RMSE = 0.6007 eV** ($R^2 = 0.8170$) on the sealed 632 test set.
   - Documentation is updated to report full test set metrics alongside the 20-sample benchmark subset metrics.
