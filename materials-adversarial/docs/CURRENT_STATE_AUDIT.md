# Current State Forensic Audit

**Audit Date:** 2026-09-12  
**Scope:** `materials-adversarial/` inside `Mrudula-itsjuzme/cyberbio`  
**Audited Checkout Path:** `/home/mrudula/Downloads/DL_cyberbio`  
**Git State:** Branch `main`, HEAD Commit `3256c9f27e8cf44bf97e92b10c62dd4ec24ec228`, Remote `https://github.com/Mrudula-itsjuzme/cyberbio.git`  
**Methodology:** Full empirical audit of code, configuration files, dataset manifests, model binaries, result artifacts, test suites, and git history.

---

## 1. Executive Verdict

1. **Repository Verification & Isolation:**
   - The active working repository was conclusively located at `/home/mrudula/Downloads/DL_cyberbio`.
   - All required directories (`materials-adversarial/`, `src/materials_adv/`, `scripts/`, `tests/`, `configs/`, `results/`, `docs/`) exist and are populated.
   - No other local checkouts of `cyberbio` exist on the system.

2. **Experimental Lineage Coexistence:**
   - The repository contains **two distinct experimental lineages**:
     1. **Historical OpenPoly Tg Lineage (K, 247 usable rows):** High-temperature glass transition experiments evaluated on a tiny 6-sample test split. Models in `results/models/transformer_regressor_residualized`, `results/models/transformer_regressor_mcmc_defended`, `results/models/transformer_defended_phase2b`, and `results/phase2c/`.
     2. **Active polyVERSE Bandgap Lineage (eV, 4,209 usable rows):** Bandgap prediction experiments evaluated on a 632-sample test split. Baseline and defended models in `results/models/transformer_regressor` and `results/models/transformer_defended`.
   - **Critical Rule:** Results must NEVER be merged across these two lineages. The active research lineage is **polyVERSE Bandgap (eV)**.

3. **Scientific & Technical Audit Findings:**
   - **Test Suite Status:** Running `./scripts/run_tests.sh` yields **241 passed, 1 skipped, 0 failed**. Direct system `pytest` invocation fails due to an environment conflict with system ROS 2 plugins (`launch_testing` missing `lark`); the wrapper script `scripts/run_tests.sh` correctly resolves this.
   - **Label Preservation Flaw:** Attacks that modify chemical formula or connectivity (`substitution`, `insertion`, `deletion`, `rearrangement`, `probabilistic_mcmc`) alter the physical property. Assuming they preserve the measured bandgap during adversarial augmentation is scientifically **UNVERIFIED / INVALID**. Only RDKit SMILES randomization represents true molecular graph invariance.
   - **Evaluation Candidate Unpairing:** Re-evaluating the headline Phase 2 result files revealed that baseline and defended models were evaluated on **unpaired candidate sets** for 5 of 6 attack families, confounding direct causal robustness comparisons.
   - **"MCMC" Overstatement:** The `probabilistic_mcmc` attack is a model-guided stochastic search, lacking Metropolis-Hastings proposal ratio correction, stationary target density, and convergence diagnostics.

---

## 2. Lineage Matrix

| Feature | Historical Lineage (OpenPoly Tg) | Active Lineage (polyVERSE Bandgap) |
| :--- | :--- | :--- |
| **Dataset Source** | OpenPoly / Literature CSV | polyVERSE (`data/raw/bandgap_chain.csv`) |
| **Target Property** | Glass Transition Temperature ($T_g$) | Bandgap ($E_g$) |
| **Target Units** | Kelvin ($K$) | Electron-Volts ($eV$) |
| **Usable Sample Count** | 247 usable rows (443 non-null Tg in raw) | 4,209 usable rows |
| **Split Strategy** | Scaffold Split (204 Train / 37 Val / 6 Test) | Deterministic Random Split (2,946 Train / 631 Val / 632 Test) |
| **Split Artifact** | Legacy embedded splits | `data/processed/splits.json` |
| **Primary Checkpoints** | `transformer_regressor_residualized`, `phase2c/*` | `transformer_regressor`, `transformer_defended` |
| **Baseline Test MAE** | $50.62 - 79.99\ K$ | $0.4619\ eV$ |
| **Status** | HISTORICAL / INACTIVE | ACTIVE |

---

## 3. Dataset and Split Audit

For the active **polyVERSE Bandgap** lineage:
- **Source Dataset:** `data/raw/bandgap_chain.csv` $\rightarrow$ `data/processed/processed.csv`
- **Representation Column:** `smiles` (raw) $\rightarrow$ `original_representation` (processed)
- **Target Column:** `bandgap_chain` (raw) $\rightarrow$ `property_value` (processed)
- **Units:** `eV`
- **Total Usable Records:** 4,209
- **Train Count:** 2,946 (70.0%)
- **Validation Count:** 631 (14.99%)
- **Test Count:** 632 (15.01%)
- **Split Seed:** `20260815` (scaffold/random mix, materialized in `data/processed/splits.json`, `test_sealed: True`)

### Verification Checks
- **Index Overlap:** 0 overlapping indices between Train, Val, and Test splits.
- **Canonical Structure Leakage:** 0 canonical SMILES overlap between Train, Val, and Test splits.
- **Train/Test Contamination:** None detected.
- **Target Scaler Provenance:** Standard Scaler (Mean: `4.474831`, Std: `1.456305`) fit strictly on the 2,946 training records.
- **Tokenizer & Vocabulary:** 36-token chemical PSMILES vocabulary stored in `data/processed/vocab.json`.
- **Maximum Sequence Length:** 256 tokens.

---

## 4. Model and Checkpoint Inventory

| Checkpoint Directory | Lineage | Target | Scaler Mean / Std | Train / Val / Test N | Test MAE | Provenance & Clean/Defended Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `results/models/transformer_regressor` | polyVERSE | Bandgap (eV) | 4.4748 / 1.4563 | 2946 / 631 / 632 | 0.4619 eV | COMPLETE — Active Baseline Model |
| `results/models/transformer_defended` | polyVERSE | Bandgap (eV) | 4.4748 / 1.4563 | 14246 / 631 / 632 | 0.4601 eV | COMPLETE — Active Defended Model |
| `results/models/transformer_defended_phase2b` | OpenPoly | $T_g$ (K) | 330.71 / 88.046 | 1286 / 37 / 6 | 50.64 K | HISTORICAL — Tg Defended |
| `results/models/transformer_defended_phase2b_balanced` | OpenPoly | $T_g$ (K) | 330.71 / 88.046 | 700 / 37 / 6 | 68.56 K | HISTORICAL — Tg Defended Balanced |
| `results/models/transformer_regressor_mcmc_defended` | OpenPoly | $T_g$ (K) | 330.71 / 88.046 | 402 / 37 / 6 | 48.13 K | HISTORICAL — Tg MCMC Defended |
| `results/models/transformer_regressor_residualized` | OpenPoly | $T_g$ (K) | -1.08e-13 / 83.58 | 204 / 37 / 6 | 50.62 K | HISTORICAL — Tg Residualized |
| `results/phase2c/baseline_seed*` (5 seeds) | OpenPoly | $T_g$ (K) | 330.71 / 88.046 | 204 / 37 / 6 | 46.12–79.99 K | HISTORICAL — Tg Multi-seed Baseline |
| `results/phase2c/defended_seed*` (5 seeds) | OpenPoly | $T_g$ (K) | 330.71 / 88.046 | 1286 / 37 / 6 | 34.49–77.57 K | HISTORICAL — Tg Multi-seed Defended |
| `results/phase4/closed_loop_*` | polyVERSE | Bandgap (eV) | 4.4748 / 1.4563 | 5762–13723 / 631 / 632 | 0.4417–0.4533 eV | COMPLETE — Active Phase 4 Closed-Loop Runs |

---

## 5. Attack Inventory

| Attack Name | Code Path | Registered | Tested | Executed | Black-box / White-box | Target Label Preserved? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Substitution** | `src/materials_adv/attacks/substitution.py` | YES | YES | YES | Black-box (vocab pool) | **NO** (Alters chemical formula/atoms) |
| **Insertion** | `src/materials_adv/attacks/insertion.py` | YES | YES | YES | Black-box (token insertion) | **NO** (Alters chemical structure) |
| **Deletion** | `src/materials_adv/attacks/deletion.py` | YES | YES | YES | Black-box (token deletion) | **NO** (Alters chemical structure) |
| **Rearrangement** | `src/materials_adv/attacks/rearrangement.py` | YES | YES | YES | Black-box (window swap) | **NO** (Alters connectivity/isomerism) |
| **Reordering** (Legacy) | `src/materials_adv/attacks/reordering.py` | YES | YES | Deprecated | Black-box (pairwise swap) | **NO** (Superseded by rearrangement) |
| **RDKit Randomization** | `src/materials_adv/attacks/randomization.py` | YES | YES | YES | Black-box (RDKit SMILES) | **YES** (Preserves molecular graph) |
| **Probabilistic MCMC** | `src/materials_adv/attacks/probabilistic.py` | YES | YES | YES | White-box (model loss/drift) | **NO** (Stochastic edit proposals) |
| **Search Strategies** | `src/materials_adv/attacks/search/` | N/A | YES | YES | Random/Greedy/Metropolis | **NO** (Search wrapper over edits) |

---

## 6. Defense Inventory

| Defense Method | Implemented | Tested | Executed | Result Available | Scientifically Validated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adversarial Augmentation** | YES | YES | YES | `augmented_train.csv` | **PARTIAL** (Assumes label preservation for chemistry-changing edits) |
| **Fixed Scaler Control** | YES | YES | YES | `metrics.json` | **YES** (Frozen baseline scaler prevents artificial drift) |
| **Defended Training** | YES | YES | YES | `transformer_defended` | **PARTIAL** (Effective for substitutions, unverified for general robustness) |
| **Randomized SMILES Training** | YES | YES | YES | Phase 4 Closed-Loop | **YES** (Representation-invariance training) |
| **MCMC / Mixed Training** | YES | YES | YES | Phase 4 Closed-Loop | **PARTIAL** (Relies on model teacher labels) |
| **Re-attack Evaluation** | YES | YES | YES | `phase2_test_defended.jsonl` | **PARTIAL** (Unpaired candidates in historical Phase 2 files) |
| **Multi-seed Evaluation** | YES | YES | YES | `results/phase2c/` (Tg) | **PARTIAL** (Executed for Tg, not yet for Bandgap) |

---

## 7. Results Directory Forensics

| Directory / Artifact | Lineage | Phase | Complete? | Canonical? | Trustworthy? | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `results/models/transformer_regressor/` | polyVERSE | Baseline | YES | YES | YES | Clean baseline Bandgap model (MAE 0.4619 eV) |
| `results/models/transformer_defended/` | polyVERSE | Phase 2 | YES | YES | PARTIAL | Defended Bandgap model; evaluated on unpaired test candidates |
| `results/models/transformer_*_phase2b*` | OpenPoly | Phase 2B | YES | No (Historical) | NO | Historical Tg model on 6-sample test set |
| `results/phase2/` | polyVERSE | Phase 2 | YES | No (Superseded) | PARTIAL | Attack JSONL records with unpaired baseline/defended candidates |
| `results/phase2c/` | OpenPoly | Phase 2C | YES | No (Historical) | NO | 5-seed Tg models on 6-sample test set |
| `results/phase4/` | polyVERSE | Phase 4 | YES | YES | YES | Closed-loop randomization & teacher-labeled evaluation |
| `results/representation_attribution*/` | polyVERSE | Phase 5 | YES | YES | YES | Token occlusion & permutation sensitivity attribution |
| `results/attack_efficiency*/` | polyVERSE | Phase 6 | YES | YES | YES | Query budget efficiency curves across search strategies |

---

## 8. Documentation Contradictions & Audit

1. **Split Strategy Discrepancy:**
   - `README.md` states: "Stratified 80/10/10 split".
   - `configs/dataset.yaml` states: `strategy: random`, `train_frac: 0.70`, `val_frac: 0.15`, `test_frac: 0.15`.
   - `data/processed/splits.json` actual counts: 2,946 / 631 / 632 (70 / 15 / 15).
   - **Verdict:** README claim is CONTRADICTED.

2. **Phase 1 Attack Headline Claims:**
   - `README.md` claims 4,232 successful attacks with max drift 5.47 eV.
   - Checked-in artifact `results/phase2/baseline_adversarial_results.jsonl` recomputed values: 4,253 candidates exceeding 0.4619 eV drift; maximum drift is 5.589 eV.
   - **Verdict:** README headline numbers are CONTRADICTED by actual checked-in JSONL files.

3. **Candidate Pairing Claim:**
   - `docs/FINAL_RESULTS.md` claims baseline and defended models were evaluated on identical candidate sets.
   - Recomputing candidate strings across the JSONL files proves candidate sets are **UNPAIRED** for substitution, insertion, deletion, rearrangement, and MCMC.
   - **Verdict:** Claim is CONTRADICTED.

---

## 9. Code Health & Environment Audit

- **Environment & Pytest Wrapper:** Direct invocation of `pytest` fails on Linux systems with ROS 2 installed due to `/opt/ros/jazzy` exporting `launch_testing` without `lark`. Executing `./scripts/run_tests.sh` (which sets `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=""`) bypasses ROS 2 and succeeds cleanly.
- **RDKit Seeding:** `Chem.MolToSmiles(..., doRandom=True)` uses internal RDKit C++ RNG, which is not seeded by NumPy `np.random.Generator`.
- **Probabilistic Search Algorithmic Rigor:** The `probabilistic_mcmc` module implements a stochastic local search without Hastings correction for proposal asymmetry. It should be described as model-guided stochastic search rather than Markov Chain Monte Carlo sampling.

---

## 10. Test Suite Execution Results

Running `./scripts/run_tests.sh`:
- **Total Collected:** 242 tests
- **Passed:** 241 tests
- **Skipped:** 1 test (`test_validation.py::test_rdkit_validity_skipped_if_rdkit_missing`, skipped because RDKit is present)
- **Failed:** 0 tests
- **`git diff --check`:** Clean (0 formatting or whitespace warnings)

---

## 11. Pipeline Summary & Status

### Exact Current Pipeline
1. **Data Preprocessing & Splitting:** `data/raw/bandgap_chain.csv` $\rightarrow$ `data/processed/processed.csv` $\rightarrow$ `splits.json` (70/15/15).
2. **Vocabulary:** `data/processed/vocab.json` (36 tokens).
3. **Baseline Training:** `src/materials_adv/training/train.py` $\rightarrow$ `results/models/transformer_regressor/model.pt` + `scaler.json`.
4. **Attack Generation:** `src/materials_adv/attacks/run_attacks.py` / `src/materials_adv/experiments/pipeline.py`.
5. **Defended Training:** `augmented_train.csv` + frozen baseline scaler $\rightarrow$ `results/models/transformer_defended/model.pt`.
6. **Closed-Loop & Attribution:** `scripts/run_closed_loop.py`, `scripts/run_representation_attribution.py`, `scripts/run_attack_efficiency.py`.

---

## 12. Assessment & Safe Next Steps

- **Implemented but Not Executed:** Multi-seed bandgap defended evaluation; external cross-dataset generalization.
- **Executed but Not Scientifically Trustworthy:** Historical Tg 6-sample test results; label-preserving claims for chemistry-changing attacks; defense comparisons from unpaired Phase 2 candidate sets.
- **Safe to Build On:** Core Transformer architecture, dataset preprocessing & 70/15/15 splits, standardized pipeline runner (`src/materials_adv/experiments/pipeline.py`), attack edit generators, and test suite.
- **Is Repository Safe to Continue From? YES**, provided that Phase 2 strictly builds on the polyVERSE Bandgap lineage, enforces paired candidate evaluation, and uses teacher labels for target-changing perturbations.
