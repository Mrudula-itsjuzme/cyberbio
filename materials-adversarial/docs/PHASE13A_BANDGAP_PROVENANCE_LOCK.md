> [!NOTE]
> **Status**: CANONICAL

# Phase 13A: Exact Bandgap Target Provenance Lock

## Objective
To strictly trace and lock the provenance of the `data/raw/bandgap_chain.csv` target values used in this project, clearly demarcating verified exact-dataset facts from hypothesized or general literature context.

## Provenance Analysis

The raw dataset used across Phase 1 to Phase 12B originates directly from the polyVERSE repository.

### Exact Source File
- **Source:** `Ramprasad-Group/polyVERSE` GitHub Repository.
- **Repository Path:** `Other/bandgap_chain.csv`
- **Zenodo DOI:** `10.5281/zenodo.13352644`
- **Rows:** 4,209

### Property Semantics
The target column `bandgap_chain` refers to the electronic bandgap of 1D periodic polymer chains. PolyVERSE explicitly categorizes this file as part of its machine-learning-ready datasets derived from Density Functional Theory (DFT) calculations.

## Categorized Provenance Claims

Every claim regarding the generation of this dataset must carry an evidence level:
- **A. Verified Exact-Dataset Fact**: Directly supported by metadata/source for `bandgap_chain.csv`.
- **B. Related Literature Context**: Supported by a Ramprasad polymer paper but not proven to describe this exact file.
- **C. Unknown**: Not recoverable from available evidence.

| Claim | Evidence | Evidence Level | Canonical Status |
| :--- | :--- | :---: | :---: |
| Data originates from polyVERSE | Matched github repository path `Other/bandgap_chain.csv` | A | Verified |
| Target is Computational DFT | polyVERSE documentation | A | Verified |
| Calculated with HSE06 | General Ramprasad group literature | B | Unverified |
| Calculated with PBE | General Ramprasad group literature | B | Unverified |
| Exact functional and basis set | None in local file or polyVERSE release notes | C | Unknown |

## Conclusion and Oracle Implications

Since the exact functional (e.g., PBE vs. HSE06) and basis set (e.g., PAW) used to generate `bandgap_chain.csv` are **UNKNOWN**, we cannot explicitly dictate the parameterization of a future "Matched Oracle". Instead, any future oracle deployment must begin with an empirical calibration phase against a deterministic `calibration_manifest.csv` to scientifically justify its acceptability via standard error metrics.
