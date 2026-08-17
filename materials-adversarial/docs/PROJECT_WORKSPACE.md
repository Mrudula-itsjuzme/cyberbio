# PROJECT WORKSPACE

Living source of truth for this project. Update on every research decision,
milestone, experiment, bug, architecture change, dataset finding or result.

**Never fabricate results.** Uncertain items are labelled `PENDING`, `UNKNOWN`
or `NEEDS VERIFICATION`. Decisions are appended to §14, never silently rewritten.

Last updated: 2026-08-15

---

## 1. Project Overview

**Title:** Learning to Attack and Defend: A Unified Adversarial Framework for
Robust Materials Sequence Modelling

Eventual pipeline:

```
representation -> target model -> attack generator -> candidates -> validity
filter -> adversarial examples -> drift evaluation -> adversarial training ->
defended model -> re-attack -> robustness evaluation
```

**Phase 1 (current) is the attack side only.** Defense, MCMC, probabilistic
attacks and uncertainty are out of scope and deliberately unimplemented.

---

## 2. Current Status

| Component | Status |
|---|---|
| Repository structure | DONE |
| Tokenizer (rules) | DONE — 176 tests passing |
| Token space + `editable_positions` | DONE |
| Attacks: deletion, insertion, reordering | DONE |
| Attack: substitution | Mechanics DONE, replacement pool **PENDING(dataset)** |
| Attack generator pipeline | DONE — runs end-to-end |
| Validation (representation + plausibility) | DONE — RDKit active |
| Records / metrics / attack metrics | DONE |
| Dataset audit tooling | DONE — verified against synthetic fixtures |
| Configs | DONE |
| **Dataset** | **DONE — PRESENT** |
| Preprocessing / splits / model / training | **IN PROGRESS** |
| Experiment 1 (baseline) | NOT STARTED |
| Experiments 2–5 (attacks) | NOT STARTED |

**Nothing has been trained. No results exist.**

---

## 3. Research Decisions

### Confirmed

| Decision | Value | Rationale |
|---|---|---|
| Domain | Polymer property prediction | Project brief |
| Representation | PSMILES | Project brief |
| Target property | Glass transition temperature (Tg) | Project brief |
| Task | Regression | Project brief |
| Target model | Small Transformer encoder | Controlled attack target, not SOTA |
| Attacks | Substitution, insertion, deletion, local rearrangement | Project brief |
| Primary metric | `prediction_drift` | Project brief |
| Drift storage | **SIGNED** (`abs()` at reporting) | Direction of Tg shift is a real finding; `abs()` discards it |
| Attack granularity | **Token level, never character level** | Character edits split `Cl` and corrupt `[C@@H]` — a string bug masquerading as chemistry |
| Validity semantics | **Strict UNCHECKED** | Syntax alone never implies chemical validity |
| Split strategy | Scaffold (primary), random (secondary comparison) | Random alone inflates performance via structural near-duplicates |
| Test set | **SEALED** | Attacks developed against validation only |
| RDKit | Installed (2026.03.5) | Canonicalization, dedup, representation validity |
| torch | **Deferred** | Hyperparameters need the audit; no GPU here |
| Record schema | Phase 2+ fields present now as null | Prevents mid-project schema migration breaking comparability |

### PENDING

- Exact Tg column name — `Tg (K)`
- Tg units (K vs °C) — Assumed K (values 146-589)
- Usable Tg record count — **443 non-null records**
- Duplicate-conflict policy (mean / keep-first / drop) — **Drop** chosen (31 conflicts removed)
- All model hyperparameters — Configured for small Transformer
- Substitution replacement pool — will derive from training-split vocabulary
- `success_criterion.min_abs_drift` — should be calibrated to baseline test MAE

---

## 4. Dataset

### Status: **PRESENT — AUDITED**

Dataset obtained via `curl` from the OpenPoly benchmark GitHub repository.
The test `tests/test_audit_script.py::test_real_data_dir_is_currently_empty` may now fail or need adjustment.

### Source (identified 2026-08-15)

- **Paper:** Wang et al., *"OpenPoly: A Polymer Database Empowering Benchmarking
  and Multi-property Predictions"*, Chinese Journal of Polymer Science (2025).
  DOI `10.1007/s10118-025-3402-y`. **Paywalled — schema UNVERIFIED.**
- **Data/code:** https://github.com/WangGroupFDU/Openpoly_benchmark
  - `data/final_polymer_properties_fromliterature.csv`
  - `data/final_polymer_property_counts_fromliterature.csv`
- **Portal:** `cleanenergymaterials.cn/polymer/polymer_database/experiment_polymer_database`
- Pretrained weights (not needed for us): Zenodo `10.5281/zenodo.15551637`

### Published figures — NEEDS VERIFICATION against actual files

~3,985 curated polymer–property pairs across **26 properties**, covering ~745
polymers; raw ~284,902 entries.

### ⚠ RESEARCH RISK: probable Tg scarcity

3,985 pairs spread over 26 properties means the **Tg slice is a fraction of
3,985**, and unique polymers are bounded by ~745. Corroborating evidence: the
paper reports **XGBoost outperforming deep learning** on Tg (R² 0.65–0.87)
precisely because of data scarcity.

No sample-size threshold is being asserted as a rule. Capacity will be chosen
from the **measured** count after the audit. If the slice is small, the honest
framing is a deliberately small Transformer serving as a **controlled attack
target**, not a competitive predictor — legitimate for this research question,
but it must be stated explicitly in any writeup rather than discovered later.

### Audit plan (tooling ready, awaiting data)

`scripts/audit_dataset.py` **discovers** schema; it never assumes column names.
It proposes candidates with evidence and `confirmed: false`, exits non-zero on
ambiguity, and never auto-writes `configs/dataset.yaml`.

Notable method choice — **Tg unit detection**: a naive largest-gap bimodality
probe was tested against a synthetic 50/50 °C+K mixture and **failed** (spurious
gap at −202; the components overlap). The script instead uses (a) negative values,
which conclusively rule out pure Kelvin, and (b) a **shift-correlation** test
against a 273.15 offset, plus a check for one polymer carrying two values
differing by ≈273.15. Verdict is always `requires_human_confirmation: true`.

### Tokenizer round-trip caveats (§9 of brief)

`detokenize(tokenize(x)) == x` holds **exactly** when tokenization completes
without unknown characters. Broken by:
1. `encode`→`decode` instead — OOV tokens map to `<unk>`, unrecoverable
2. Upstream RDKit canonicalization changing the string before tokenizing
3. `on_unknown="unk"` mode
4. Special tokens — identity holds only after stripping

No normalization or case folding is applied (`C` aliphatic vs `c` aromatic).

---

## 5. Architecture

```
PSMILES -> chemical tokenizer -> token ids -> embedding
        -> Transformer encoder -> pooling -> regression head -> Tg
```

Attacks see the model **only** through `PredictorProtocol`
(`predict(list[str]) -> np.ndarray`, plus `target_units`). Keeping that seam
narrow is what lets gradient-based or probabilistic attacks be added later
without touching the evaluator. `ConstantPredictor` implements the same protocol,
so the whole pipeline is exercisable without torch.

Three concepts kept structurally separate (§11 of brief):
1. **Representation validity** — `validation/representation.py`
2. **Chemical plausibility** — `validation/plausibility.py`
3. **Adversarial effectiveness** — `evaluation/attack_metrics.py`

---

## 6. Code Structure

```
configs/       dataset.yaml model.yaml tokenizer.yaml attack.yaml
data/          raw/ interim/ processed/ attacks/         (raw/ EMPTY - blocked)
src/materials_adv/
  data/        tokenizer.py(DONE) loader.py preprocessing.py splits.py(partial)
  models/      registry.py(DONE) transformer.py regression.py    (PENDING)
  attacks/     token_space.py base.py registry.py generator.py   (DONE)
               deletion.py insertion.py reordering.py            (DONE)
               substitution.py            (mechanics DONE, pool PENDING)
  validation/  representation.py plausibility.py pipeline.py     (DONE)
  evaluation/  records.py metrics.py attack_metrics.py           (DONE)
  training/    train.py evaluate.py                              (PENDING)
  utils/       seeding.py logging.py io.py config.py optional.py pending.py
scripts/       audit_dataset.py run_tests.sh
tests/         5 files, 176 tests
```

Blocked entry points raise `PendingImplementation(what, blocked_on,
unblocks_when)` rather than returning a plausible wrong answer.

---

## 7. Attack Generator

Common interface: `generate(tokens, n_variants) -> list[AttackOutcome]` and
`metadata()`. Registry-based, so a new attack is one file plus one decorator.

**`editable_positions` — the key safety primitive.** Some edits *guarantee*
invalidity: deleting one paren of a pair, one digit of a ring-closure pair, or an
attachment point. An "attack" built from those would measure the validity
filter, not model robustness. Such positions are protected by default, and
protection is a **config flag** so relaxing it stays measurable rather than
assumed.

**Why substitution's pool is PENDING:** deletion/insertion/reordering are
*closed* operations needing only tokens already present. Substitution must answer
"replace with what?" A hardcoded list would inject the author's chemical priors
into the headline result, and a pool not matching the training distribution would
measure out-of-distribution handling — a different claim, easily mistaken for the
intended one.

`number_of_changes` is **computed** from the token diff, not trusted from the
attack's own bookkeeping, so a buggy attack cannot under-report perturbation size.

---

## 8. Experiments

| # | Experiment | Status |
|---|---|---|
| 1 | Clean baseline | DONE — test MAE 52.02 K |
| 2 | Substitution attacks | DONE — Phase 1D |
| 3 | Insertion attacks | DONE — Phase 1E |
| 4 | Deletion attacks | DONE — Phase 1E |
| 5 | Local rearrangement | DONE — Phase 1E |
| 6 | Attack comparison / attribution | DONE — Phase 1F |
| — | Adversarial training (2A) | DONE — clean result later found confounded |
| — | **Controlled defense ablation (2B)** | DONE — single seed; clean gain did not survive the scaler control |
| — | **Multi-seed confirmation (2C)** | **DONE — 2B's large robustness effects do NOT replicate; only worst-case (max) drift reduction is consistent** |
| 7–11 | Multi-step, probabilistic, MCMC, re-attack | NOT STARTED — Phase 3, on hold |

---

## 9. Results

**No research results exist.** Nothing has been trained.

The only recorded numbers are scaffold verification, not findings:

- 176 tests pass, 1 skipped (RDKit-absent policy test, correctly inactive)
- Tokenizer round-trips 14/14 curated PSMILES exactly
- Pipeline smoke test: 20 candidates from 2 synthetic polymers using
  `ConstantPredictor`. RDKit marked **8 valid / 12 invalid**. Drift is 0.0 by
  construction (constant predictor), so `n_success: 0` is expected, not a finding.

⚠ **Observation to carry forward:** that ~60% invalid rate on unconstrained
attacks means the validity filter will materially shape Experiments 2–5. Drift
must be reported **conditioned on validity**, and per-attack validity rates
reported alongside drift, or the headline number will silently confound the two.

---

## 10. Problems and Bugs

| # | Issue | Status |
|---|---|---|
| 1 | **OpenPoly dataset absent.** No polymer data anywhere on the machine. | **RESOLVED** — supplied 2026-08-15, audited, 247 usable Tg records |
| 2 | Paper paywalled; schema unverifiable without the files. | OPEN — mitigated by discovery-based audit |
| 3 | pandas 3.0 uses a `str` dtype, not `object`. An `== object` check found **zero** text columns, so the audit would have silently reported "no representation column" for a good file. | **FIXED** — `_is_text_dtype()` handles both; regression test added |
| 4 | Tokenizer regex ordering: single-char class before `Br\|Cl` turns `Cl` into `C` + stray `l`. | **PREVENTED** — ordering pinned by regression tests |
| 5 | Naive largest-gap bimodality probe failed on a synthetic 50/50 °C+K mixture. | **FIXED** — replaced with shift-correlation |
| 6 | System ROS 2 install exports a global `PYTHONPATH`; pytest autoloads its plugins, which crash on missing `lark`. | **WORKED AROUND** — `scripts/run_tests.sh`. Environment quirk, not a project bug |
| 7 | No GPU; ~3.5Gi RAM free. | KNOWN — CPU-only; constrains model size |
| 8 | **Scaler refit on augmented data (Phase 2A confounder).** `train()` refit `TargetScaler` on whatever frame it received, shifting the normalization mean by -10.09 K between Phase 1 and 2A and making the clean-MAE comparison uninterpretable. | **CONTROLLED** — Phase 2B adds `scaler_path` to load a fixed scaler; confounder quantified at ~76% of the reported gain |
| 9 | **`train()` overwrote `configs/model.yaml:baseline_metrics` on every run.** After Phase 2A the config held the *defended* model's metrics, so `run_attacks.py` silently used 46.20 K instead of 52.02 K as its drift threshold. | **FIXED** — runs write `metrics.json` beside their own checkpoint; `write_back_config=False`; threshold passed explicitly via `--threshold` |
| 10 | **Undetected duplicate class.** 19 "adversarial" training strings were byte-identical to clean training strings — sample-weight inflation, not augmentation. Distinct from the 11 adv-vs-adv duplicates reported in 2A. | **FIXED in 2B** — both classes removed and reported separately |
| 11 | **Stale scaffold-era tests.** 6 tests in `test_attacks.py` / `test_scaffold_integrity.py` / `test_audit_script.py` assert the pre-data blocked state (`TokenRole.ATOM`, stubs raising, configs null, `data/raw` empty). Phases 1–2A intentionally resolved all of these. | **FIXED in 2C** — all 6 updated to the current contract, none deleted; 200 tests pass |
| 12 | **`@register_attack` dropped from three attacks.** The Phase 1D/1E rewrites of insertion, deletion and rearrangement lost their registry decorators, leaving only `substitution` registered — `build_attack("deletion")` raised `KeyError`. Experiments construct attacks directly, so no result is affected, but the pluggability contract was silently broken. | **FIXED in 2C** — decorators restored; two tests pin all four registered and buildable |
| 13 | **Two API inconsistencies** (noted, not fixed): `RearrangementAttack` lost its `window_size >= 2` validation and degrades to zero candidates instead of raising; substitution guards a missing pool with `PendingImplementation` while insertion uses a required positional argument. | **OPEN — cosmetic.** Both behaviours are safe; tests pin the actual behaviour rather than an aspirational one |
| 14 | **`reordering.py` is dead code.** Superseded by `rearrangement.py`, which all experiments use. Still imports and registers itself. | **OPEN — cosmetic.** Left in place rather than deleted mid-experiment; candidate for removal before publication |

---

## 11. Baseline Sanity Gate (Run 2026-08-15)

### Experiment 1A: Unnormalized baseline / FAILED SANITY GATE
- **Training behavior:** Initial loss was extremely high (115,450). Validation MAE dropped from 379 to 240, but predictions effectively collapsed toward a constant ~143 K.
- **Identified issues:** The model is predicting a near-constant ~143 K for all inputs. This is because the raw regression target (Tg in Kelvin, mean ~342, std ~92) is completely un-normalized. The network spends 50 epochs just trying to shift its initial zero-bias output upwards, plateauing prematurely.
- **Transformer metrics:** Test MAE: 319.37, RMSE: 320.89, R2: -105.25.
- **Is the baseline READY FOR ATTACKS?** **NO.** 

### Experiment 1B: Normalized baseline (TargetScaler) / PASSED SANITY GATE
- **Actual dataset size:** 741 raw rows -> 247 usable final samples
- **Split sizes:** Train: 204, Val: 37, Test: 6 (scaffold groups)
- **Model configuration:** Transformer (d_model=64, n_layers=2, n_heads=4), Batch size=16, lr=1e-3, epochs=50. Normalization fitted on training targets only.
- **Parameter count:** 86,337 parameters
- **Trivial baseline metrics:** Mean Predictor MAE: 132.54, Median Predictor MAE: 130.67
- **Transformer metrics:** Test MAE: 52.02, RMSE: 71.30, R2: -4.25
- **Prediction behavior:** True: 423.00 K -> Pred: 430.23 K. Range of predictions is ~334.93 K to ~552.60 K.
- **Training behavior:** Initial normalized training MSE was 0.8053. Validation MAE (in Kelvin) started at 84.07 K and smoothly converged to ~79 K before early stopping triggered at epoch 22.
- **Is the baseline READY FOR ATTACKS?** **YES.** The collapse is resolved. The baseline demonstrates sensitivity to structural inputs, no longer collapses, and significantly outperforms the Mean/Median trivial predictors.

### Experiment 1D: Role-Constrained Substitution Attack (Run 2026-08-15)
**Status:** "Token-Role Constrained Substitution Baseline"

- **Attack Generator:** `SubstitutionAttack` with `n_edits=1`, configured via vocabulary pool derived purely from training data (45 tokens).
- **Refined Taxonomy:** During a preliminary smoke test, it was discovered that unconstrained role matching permitted substitutions like `= -> .` and `C -> n`. The `TokenRole` taxonomy was refined to strictly separate `ALIPHATIC_ATOM` from `AROMATIC_ATOM`, and `BOND` from `DISCONNECT` (`.`). The attack is now constrained such that it only substitutes within these strict categories.
- **Split Used:** Validation split exclusively (37 scaffold groups). 
- **Model Used:** TransformerRegressor (Exp 1B checkpoint).
- **Candidate Generation:** 181 raw candidates (5 variants per validation sample).
- **Validity:** 94 (51.9%) representation-valid (which yielded 92 unique canonical structures). 87 (48.1%) were invalid. Note that the valid candidate rate increased due to the tightened taxonomy constraints.
- **Invalidity Reasons:** The RDKit parse failures remain dominated by explicit valence violations (e.g., substituting an aliphatic carbon with F/Cl and breaking the bond count), and failed kekulization (breaking aromatic rings). 1 candidate was rejected for an uncommon element ('K').
- **Drift Statistics (Valid Only):** 
  - Mean drift: 9.34 K
  - Median drift: 6.24 K
  - Std Dev: 11.07 K
  - 75th %ile: 10.98 K
  - 90th %ile: 23.96 K
  - 95th %ile: 29.30 K
  - Max drift: 63.97 K
- **Attack Effectiveness:** Out of the 94 valid adversarial strings, only 2 (2.1% of valid, 1.1% of total) induced an absolute prediction drift greater than the baseline error-calibrated exploratory threshold of 52.02 K.

#### Phase 1D Limitations
- **Tiny sealed test set:** The baseline test MAE of 52.02 K comes from a 6-sample test set and is statistically fragile. It is an *exploratory threshold* and should not be treated as a universal definition of adversarial success.
- **Random substitution:** Even within tight chemical categories, the attack is largely brute-force random substitution. The low success rate reflects the difficulty of navigating chemical constraints randomly, not necessarily intrinsic model robustness.
- **Representation vs Plausibility:** RDKit validity only confirms the syntax forms a valid graph; it does NOT mean the resulting sequence is a scientifically plausible or synthetically accessible polymer. A large prediction drift on a valid string does not immediately prove a chemically meaningful adversarial attack.

---

## Phase 1E Design: Insertion, Deletion, and Local Rearrangement

### Motivation
To compare different local constrained sequence perturbation mechanisms on the same trained Transformer target. The central scientific question is whether different perturbation mechanisms produce statistically different validity decay patterns and prediction-drift distributions compared to the Phase 1D substitution baseline.

### 1. Insertion Attack
- **Definition:** Inserts an allowed token at a valid editable location.
- **Constraints:** Inserts only `ALIPHATIC_ATOM`, `AROMATIC_ATOM`, or `BOND` tokens from the global training vocabulary.
- **Protected Elements:** Structural syntax tokens (`(`, `)`, `1`, `[*]`, `.`) are protected from being inserted to prevent deliberate malformed syntax generation.
- **Candidate Generation Strategy:** Select a position adjacent to an editable token. Insert a randomly selected allowed token. 

### 2. Deletion Attack
- **Definition:** Removes exactly one eligible token from the sequence.
- **Constraints:** Only `ALIPHATIC_ATOM`, `AROMATIC_ATOM`, `BRACKET_ATOM`, and `BOND` tokens can be deleted.
- **Protected Elements:** `ATTACHMENT`, `RING_CLOSURE`, `BRANCH_OPEN`, `BRANCH_CLOSE`, and `DISCONNECT` are strictly protected. Deleting any of these immediately destroys the representation syntax (e.g., removing a branch open without the close).
- **Candidate Generation Strategy:** Randomly select `n_edits` eligible tokens and remove them, shifting the sequence left.

### 3. Local Rearrangement Attack
- **Definition:** Randomly permutes tokens within a configurable local window size (e.g., `k=3` or `k=4`).
- **Constraints:** The perturbation must be strictly bounded to the local window to test order sensitivity, rather than full-sequence shuffling.
- **Protected Elements:** If a randomly selected window contains protected elements (attachments, branches, rings, disconnects), the rearrangement must either exclude the window or pin the protected elements in place while shuffling the rest.
- **Candidate Generation Strategy:** Randomly select a local contiguous window of size `k` containing no protected tokens, and randomly permute its contents.

### Common Budget, Validation, and Metrics
- **Budget:** 5 candidate attempts per sample per attack family. Shortfalls are recorded explicitly rather than padded with duplicates.
- **Validation Pipeline:** Candidate Generation -> Deduplication -> RDKit Validation -> Plausibility Assessment -> Prediction -> Drift Calculation.
- **Metrics:** Continuous distribution of `abs(pred_adv - pred_original)`. The `52.02 K` baseline MAE serves strictly as a "baseline-error-calibrated exploratory threshold". 

### Phase 1E Results

**Comparative Experiment Execution (Run 2026-08-15):**
The four constrained attack families were evaluated against the validation split under matched conditions. The sealed test set remains untouched. The table below aggregates results. Invalid candidates were filtered by RDKit prior to model scoring.

| Attack | Perturbation Budget | Raw | Unique | Valid | Validity % | Mean Drift | Median Drift | P95 Drift | Max Drift | >52.02K / Valid | >52.02K / All |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Substitution** | 1 token substituted | 181 | 181 | 87 | 48.1% | 8.18 K | 6.16 K | 22.84 K | 51.63 K | 0.0% | 0.0% |
| **Insertion** | 1 token inserted | 185 | 185 | 97 | 52.4% | 31.09 K | 19.59 K | 96.34 K | 116.90 K | 20.6% | 10.8% |
| **Deletion** | 1 token deleted | 185 | 185 | 98 | 53.0% | 30.29 K | 21.51 K | 77.87 K | 153.92 K | 15.3% | 8.1% |
| **Rearrangement**| 3 token window permuted | 101*| 101 | 59 | 58.4% | 8.01 K | 5.92 K | 27.05 K | 30.62 K | 0.0% | 0.0% |

*\*Rearrangement produced fewer raw candidates because some short polymers or sequences dense with protected syntax lacked valid length-3 windows.*

**Interpretation against Scientific Questions:**
- **A. Highest validity rate?** Rearrangement (58.4%), followed closely by Deletion (53.0%) and Insertion (52.4%). 
- **B. Largest median drift?** Deletion (21.51 K) and Insertion (19.59 K) severely out-drift Substitution (6.16 K) and Rearrangement (5.92 K).
- **C. Largest upper-tail drift (P95/Max)?** Deletion produced the maximum single outlier (153.92 K), but Insertion had a fatter tail (P95 of 96.34 K vs 77.87 K).
- **D. Largest drift per edit?** Because Insertion and Deletion are single-token edits, their mean drift/edit matches their mean absolute drift (~30 K/edit). Rearrangement modifies 2-3 tokens per window, averaging ~4.00 K drift per edited token.
- **E/F/G. Most candidates above 52.02 K threshold?** Insertion produced the most, with 20 valid candidates breaching the threshold (20.6% of valid / 10.8% of total). Substitution and Rearrangement produced *zero*.

**Limitations & Caveats:**
- **Chemically Plausible?** No. Valid RDKit syntax does not equate to chemical plausibility. A single deletion or insertion in an organic polymer backbone often creates highly unstable or chemically nonsensical radicals or fragments that simply happen to pass generic bond-math checks.
- **Scientific Finding:** The model is significantly more sensitive to *length-changing* perturbations (Insertion/Deletion) than *length-preserving* perturbations (Substitution/Rearrangement). This is a known vulnerability in sequence models, which heavily rely on positional encoding. 

### Phase 1F: Perturbation Attribution Analysis
To determine whether the massive drift generated by Insertion and Deletion was due to the edit magnitude or the sequence-length change, an attribution analysis was performed on the `341` valid Phase 1E candidates.

**Research Question:** "Is the larger prediction drift observed for insertion/deletion primarily associated with sequence-length changes, or does it arise from differences in perturbation magnitude/composition?"

**Results:**
- **Drift by Delta Length:** Length-preserving perturbations (`delta_length=0`, n=146) resulted in a mean absolute drift of **8.11 K**. Length-changing perturbations (`delta_length!=0`, n=195) resulted in a mean absolute drift of **30.69 K**.
- **Drift by Number of Edits:** The perturbation magnitude (Levenshtein token distance) does not explain the drift. Rearrangement modified an average of 2 tokens per window but produced only **8.01 K** drift. Substitution modified 1 token and produced **8.18 K** drift. Insertion/Deletion modified exactly 1 token but produced **~30.7 K** drift. 
- **Signed Shifts:** The length-changing attacks are not strictly monotonic in one direction. While Deletion exhibited a slight negative bias (mean shift -3.91 K) and Substitution a slight positive bias (+3.58 K), the absolute magnitude overshadows the directional bias.

**Interpretation:** 
The attribution analysis conclusively shows that under the tested configuration, length-changing perturbations produce substantially larger prediction drift than length-preserving perturbations, regardless of the raw number of tokens altered. 
**Important Limitation:** This analysis establishes a strong empirical correlation with length-changing operations, but it does *not* causally establish positional encoding as the sole failure mechanism. The vulnerability could stem from downstream positional shifts, altered token composition, or altered local structural receptive fields. Positional encoding remains a strong hypothesis for future investigation.

## Phase 2A: Defense Design

### Objective
Determine whether adversarial training can reduce prediction drift while preserving clean predictive performance. The defended model will be compared directly against the Phase 1 Transformer baseline.

### Threat Model
The system faces a black-box representation-level attacker capable of applying constrained single-token local operations (Substitution, Insertion, Deletion, Rearrangement). The attacker filters for RDKit representation-validity and seeks to maximize absolute prediction drift. 

### Target-Preservation Assumption (Important Scientific Caveat)
**When is an adversarial PSMILES considered a perturbation that should preserve the original target?** 
For the purpose of this controlled machine-learning defense experiment, we assume that a single local edit (e.g., deleting an atom or inserting a bond) produces an adversarial polymer whose true physical glass transition temperature ($T_g$) is identical to the original $T_g$. 
*Scientific Limitation:* This is a conservative empirical formulation. In reality, adding an aromatic ring or breaking a structural backbone could drastically shift the true $T_g$. We are intentionally ignoring physical chemistry shifts to measure purely algorithmic robustness against representation-valid syntactic drift.

### Attack-Family Training Mixture
Because Phase 1 found Insertion and Deletion to drive the largest drift, the training data augmentation will include a balanced 25/25/25/25% mix of Substitution, Insertion, Deletion, and Rearrangement, ensuring length-changing operations form 50% of the defense surface.

### Data Separation and Leakage Prevention
- **Training:** Adversarial perturbations will be generated *exclusively* from the training split. 
- **Validation:** Validation attacks will be generated dynamically and independently against the validation set. They will not be seen during training.
- **Testing:** The test set remains completely sealed and untouched.

### Evaluation Protocol
- **Clean Evaluation:** The defended model will be scored on clean validation data reporting MAE, RMSE, and R². Robustness improvements are invalid if they severely degrade clean predictive performance.
- **Robustness Evaluation:** Fresh validation attacks will be generated. We will calculate prediction drift (mean, median, P95, max) by attack family, and specifically report `drift_reduction = baseline_drift - defended_drift` to measure the cost-benefit ratio of robustness gained per unit of clean performance lost.

### Phase 2A Results (Execution 2026-08-15)

**Clean Performance Trade-off:**
- **Baseline Test MAE:** 52.02 K 
- **Defended Test MAE:** 46.20 K (Improved by ~5.8 K)
- *Finding:* Adversarial training did not destroy clean performance. In fact, the data augmentation (label-preserving token mutations) acted as a regularizer, slightly improving the baseline clean MAE.

**Robustness Gains (Drift Reduction):**
Fresh validation attacks were run against the defended model and compared to Phase 1E Baseline drifts.

| Attack Family | Mean Drift | Mean Red. | P95 Drift | P95 Red. | Max Drift | Max Red. | >52.02K Rate |
|---|---|---|---|---|---|---|---|
| **Insertion** | 20.18 K | **-10.91 K** | 54.68 K | **-41.66 K** | 87.58 K | **-29.32 K**| 6.2% (was 20.6%) |
| **Deletion** | 19.45 K | **-10.84 K** | 60.87 K | **-17.00 K** | 81.01 K | **-72.91 K**| 9.2% (was 15.3%) |
| **Substitution**| 5.37 K | **-2.81 K** | 15.92 K | **-6.92 K** | 25.92 K | **-25.71 K**| 0.0% (was 0.0%) |
| **Rearrangement**| 7.14 K | **-0.87 K** | 18.16 K | **-8.88 K** | 47.10 K | *+16.49 K* | 0.0% (was 0.0%) |

**Conclusion:** 
Adversarial training (target-preserved representation augmentation) is highly effective at smoothing the Transformer's vulnerability to length-changing edits. Insertion/Deletion mean drifts were cut by ~35%, and extreme outliers (Max/P95 drifts) were reduced drastically (up to 72 K reduction in Max Deletion drift). 
Because clean performance was strictly preserved/improved, the "cost-benefit ratio" is entirely positive. However, the caveat regarding the true chemical viability of the target-preservation assumption remains.

### Phase 2A Evaluation Audit
An explicitly requested programmatic audit was performed to guarantee the integrity of the Phase 2A comparison against the Phase 1 Baseline. No models were retrained during this audit.

**Evaluation Protocol & Leakage Verification:**
1. **Clean Test-Set Evaluation:** Identical to baseline. Test predictions/targets were inverse-transformed to Kelvin scale. The reported 46.20 K MAE is definitively from the 6 untouched test-set scaffold groups evaluated at the end of training.
2. **Test-Set Integrity:** The defended model never saw any test samples, nor were test targets used to fit the scaler.
3. **Data Overlap:** Programmatic set intersection confirmed **0** overlapping PSMILES strings between the generated Phase 2A training augmentations and the Phase 1E validation attacks. No validation attacks leaked into training.
4. **Validation Attack Freshness:** Phase 2A robustness results were calculated from freshly generated candidates produced during inference against the `transformer_defended.pt` model.

**Training Data Construction:**
- **Attack Mixture & Counts:** The script generated 3 candidates per family per train sample. After RDKit validity filtering, the exact 1,112 added adversarial training examples were:
  - Substitution: 336 (30.2%)
  - Deletion: 335 (30.1%)
  - Insertion: 314 (28.2%)
  - Rearrangement: 127 (11.4%)
- **Duplicates:** There were 11 duplicate adversarial PSMILES strings present in the augmented training set.

**Numerical Robustness Improvements (Defended vs Baseline):**
| Attack Family | Baseline Mean Drift | Defended Mean Drift | Improvement (Reduction) |
|---|---|---|---|
| **Insertion** | 31.09 K | 20.18 K | -10.91 K |
| **Deletion** | 30.29 K | 19.45 K | -10.84 K |
| **Substitution** | 8.18 K | 5.37 K | -2.81 K |
| **Rearrangement** | 8.01 K | 7.14 K | -0.87 K |

*Max Drift Reduction:* Deletion Max Drift dropped from 153.92 K -> 81.01 K (-72.91 K).
*Threshold Outlier Reduction:* Insertion >52.02 K rate dropped from 20.6% -> 6.2%. Deletion >52.02 K rate dropped from 15.3% -> 9.2%.

**Methodological Leakage / Confounding Factors Identified:**
1. **Scaler Fitting Shift:** In Phase 2A, `TargetScaler` was fitted on the *augmented* `train_df["target"].values` rather than the *clean* training targets. Because the augmented set duplicates original $T_g$ values for adversarial strings, the statistical weight of certain samples increased, slightly altering the mean/std parameters of the scaler relative to Phase 1. This means the defended model was predicting on a slightly different normalized numerical space than the baseline model.
2. **Mixture Imbalance:** The proposed 25% balance was not exactly achieved because `Rearrangement` and `Insertion` suffered higher RDKit invalidity rates during generation, skewing the training mixture slightly toward Substitution/Deletion.

---

## Phase 2B: Controlled Defense Ablation

*Executed 2026-08-15. The Phase 2A record above is preserved unchanged.*

### Motivation

Phase 2A reported a clean test MAE improvement of 52.02 K -> 46.20 K and
attributed it to adversarial training. The Phase 2A audit identified a
confounder that makes that attribution unsafe, so Phase 2B re-runs the
experiment with normalization statistics held fixed.

### The confounder

`training/train.py` fitted `TargetScaler` on whatever training frame it was
given. Phase 1 passed the clean training split; Phase 2A passed the *augmented*
frame. The 1,112 adversarial rows inherit their parent polymer's Tg, so
frequently-attacked polymers gained statistical weight and the normalization
constants moved:

| Scaler | mean (K) | std (K) | Fitted on |
|---|---|---|---|
| Phase 1 baseline | **330.713124** | **88.046275** | 204 clean training targets |
| Phase 2A defended | 320.621346 | 87.751057 | 1,316 augmented rows |
| **Phase 2B (this phase)** | **330.713124** | **88.046275** | **loaded from Phase 1, NOT refit** |

The mean shifted by **-10.09 K**. Since MAE is reported in Kelvin after
`inverse_transform`, the baseline and defended models were decoding predictions
through different affine maps. "Adversarial training helped" and "the decoder
shifted toward the test targets" were therefore not separable.

The test targets all lie in 418.7-496.0 K, well above both scaler means, so a
downward mean shift is not neutral with respect to this particular test set.

### Experimental controls

Held identical to Phase 1: architecture, tokenizer, vocabulary, split artifact,
optimizer (AdamW), learning rate (1e-3), weight decay (1e-4), epoch budget (50),
early-stopping patience (10), seed (20260815), and evaluation procedure.

Changed, by design: adversarial augmentation of the training set only.

Implementation: `train()` gained a `scaler_path` argument. When supplied it
*loads* the scaler instead of fitting one, and logs what refitting *would* have
produced (here mean=321.470130, delta -9.24 K) so the counterfactual is recorded
rather than inferred. It also gained `write_back_config=False`, because the
Phase 2A run had overwritten `configs/model.yaml:baseline_metrics` with the
defended model's numbers. Each run now writes `metrics.json` beside its own
checkpoint, so no run can overwrite another's recorded results.

### Training-set construction

Built from the **existing** Phase 2A augmentation artifact rather than
regenerating it — regenerating would re-run the attack RNG and change which
adversarial strings exist, adding a second difference between 2A and 2B.

| Quantity | Count |
|---|---|
| Clean training samples | 204 |
| Adversarial before deduplication | 1,112 |
| Duplicates removed: adversarial vs adversarial | **11** |
| Duplicates removed: adversarial identical to a clean training string | **19** |
| **Total duplicates removed** | **30** |
| Adversarial after deduplication | **1,082** |
| Total Phase 2B training rows | 1,286 |

The second duplicate class was not reported in Phase 2A. Those 19 rows were
labelled "adversarial" but are byte-identical to clean training strings, so they
functioned purely as sample-weight inflation on already-present polymers.

Attack-family counts:

| Family | Phase 2A raw | After dedup | Share |
|---|---|---|---|
| Substitution | 336 | 335 | 31.0% |
| Insertion | 314 | 312 | 28.8% |
| Deletion | 335 | 311 | 28.7% |
| Rearrangement | 127 | 124 | 11.5% |

### On balancing the mixture

The brief preferred a balanced mixture *if achievable without changing other
conditions*. It is not. Rearrangement is validity-limited at 124 examples, so
equalising families would discard **586 of 1,082 adversarial examples (54.2%)**,
changing augmentation volume drastically — a second confounder in an experiment
whose entire purpose is removing one.

Following the brief's fallback, the deduplicated natural mixture is the
**primary** Phase 2B condition, and a balanced variant was run as a **secondary
diagnostic**. The diagnostic confirms the concern: the balanced model scores
**68.56 K** test MAE, substantially worse than both the baseline and the natural
variant. Augmentation volume, not family balance, dominates at this data scale.

### Results

**Clean evaluation** — identical validation (n=37) and sealed test (n=6) samples
throughout. Phase 1 and Phase 2A reproduce their recorded values exactly, which
validates the evaluation harness.

| Model | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|---|
| Phase 1 baseline | 79.04 | 99.38 | -0.15 | **52.02** | 71.30 | -4.25 |
| Phase 2A defended | 79.50 | 100.06 | -0.17 | **46.20** | 59.36 | -2.64 |
| **Phase 2B controlled** | **74.34** | **97.08** | **-0.10** | **50.64** | 62.09 | -2.98 |
| Phase 2B balanced *(diagnostic)* | 92.38 | 110.96 | -0.43 | 68.56 | 85.73 | -6.58 |

**Robustness** — a fresh attack set (seed 20260902, 651 candidates, 330 valid,
only 22.0% string overlap with Phase 1E) was generated and applied **identically**
to both models. Attacks are generated from sample PSMILES rather than model
gradients, so a shared seed yields an identical candidate set, making this a
*paired* comparison. Verified programmatically: `candidate_sets_identical: true`.

| Family | n | Mean drift | Median | P95 | Max | >52.02 K rate |
|---|---|---|---|---|---|---|
| Substitution | 88 | 13.0 -> **7.0** | 7.9 -> 4.6 | 39.1 -> 19.8 | 102.7 -> 32.7 | 3.4% -> **0.0%** |
| Insertion | 85 | 30.7 -> **19.3** | 19.8 -> 13.9 | 95.5 -> 53.4 | 130.7 -> 98.4 | 20.0% -> **5.9%** |
| Deletion | 94 | 30.7 -> **16.8** | 21.6 -> 13.8 | 85.3 -> 42.2 | 153.9 -> 66.8 | 17.0% -> **3.2%** |
| Rearrangement | 63 | 8.0 -> 8.0 | 5.9 -> 5.3 | 26.7 -> 21.4 | 30.6 -> **35.0** | 0.0% -> 0.0% |

*(baseline -> Phase 2B controlled)*

Pooled paired comparison over the 330 valid candidates: mean |drift| 21.64 K ->
13.16 K, a **8.48 K reduction**, with **63.9%** of individual candidates improved.

### Attribution

| Quantity | Value |
|---|---|
| Phase 2A apparent improvement (52.02 -> 46.20) | **5.81 K** |
| Phase 2B improvement surviving the control (52.02 -> 50.64) | **1.38 K** |
| Residual associated with the scaler change | **4.44 K** |
| Fraction of the 2A clean-MAE gain surviving the control | **~24%** |

**The Phase 2A clean-MAE result is NOT robust to this control.** Roughly
three-quarters of the reported clean improvement is associated with the
normalization change rather than with adversarial training. The controlled clean
improvement of 1.38 K on a 6-sample test set is well inside noise and should not
be reported as a clean-performance benefit.

**The robustness result IS robust to this control**, and is in fact stronger
under it. Drift reductions persist across every length-changing family under a
fixed scaler, a fresh attack set, and a paired design: deletion mean drift
-13.9 K, insertion -11.4 K, substitution -6.0 K, and threshold-breach rates
falling by roughly two-thirds to five-sixths. This is the defensible Phase 2
finding.

A diagnostic decoding the Phase 2A model's outputs through the Phase 1 scaler
gives 42.82 K. This is *not* a valid model score — it deliberately mismatches
training and inference normalization — but it confirms that this test set's MAE
is highly sensitive to the decoder's affine constants, which is precisely why
the control was necessary.

### Limitations

1. **The 6-sample test set dominates every clean-metric caveat.** A 1.38 K
   difference across 6 points is not a meaningful effect size. All clean MAE
   comparisons here, including Phase 2A's, are statistically fragile. Validation
   (n=37) is more trustworthy and shows the same direction: 79.04 -> 74.34 K.
2. **Negative R² throughout.** Every model, baseline included, predicts worse
   than the test mean on this test set. The model is a controlled attack target,
   not a competitive Tg predictor, exactly as scoped.
3. **Attribution is associational, not causal.** This is a single controlled
   comparison with one seed. It shows the 2A clean result does not survive the
   control; it does not prove the scaler shift *caused* the difference. Multiple
   seeds would be needed to separate the effect from run-to-run variance.
4. **Single seed.** No variance estimate across training runs. The 8.48 K pooled
   drift reduction has no confidence interval.
5. **The target-preservation assumption is inherited unchanged** from Phase 2A:
   adversarial edits are assumed not to alter true physical Tg. This remains
   chemically unjustified and bounds the physical interpretation of every
   robustness number here.
6. **The 52.02 K threshold is an exploratory reference**, derived from the
   fragile 6-sample baseline MAE. It is retained only for comparability with
   Phase 1E, not as a principled definition of attack success.
7. **Deduplication and scaler control were changed together** relative to 2A.
   The 30 removed duplicates are ~2.8% of the augmented rows, so their effect is
   likely small, but it is not separately isolated.

---

## Phase 2C: Multi-Seed Robustness Confirmation

*Executed 2026-08-15. Phase 2A and 2B records preserved unchanged.*

### Objective

Determine whether the Phase 2B robustness improvement is reproducible across
random seeds. Phase 2B was a single-seed result with no variance estimate.

### Setup

Identical to the Phase 2B control in every respect: Phase 1 `TargetScaler`
frozen (mean 330.713124, std 88.046275, loaded not refit — verified identical in
all 10 runs), same tokenizer, same scaffold split artifact, same architecture,
same optimizer settings, same adversarial training set
(`train_aug_phase2b.csv`, 1,286 rows), same attack budgets.

**5 seeds x 2 conditions = 10 independent trainings**: 20260815–20260819.
`train()` gained a `seed` argument controlling weight init, DataLoader shuffling
and dropout masks. Baseline runs confirm `n_train=204` (clean only); defended
runs use the 1,286-row augmented set.

Attacks use a per-seed fresh attack seed, so each seed sees a *different*
candidate set, while baseline and defended **within** a seed see an *identical*
one. Verified programmatically: `matched_candidate_sets_all_seeds: true`.

**The test set was never loaded by this phase.** All reporting is on validation
(n=37); model selection used validation MAE only.

### Clean validation results

| Seed | Baseline MAE | Defended MAE | Δ | Baseline R² | Defended R² |
|---|---|---|---|---|---|
| 20260815 | 79.04 | 74.34 | **−4.71** | −0.15 | −0.10 |
| 20260816 | 79.06 | 79.03 | −0.03 | 0.04 | 0.05 |
| 20260817 | 78.79 | 83.84 | **+5.05** | −0.07 | −0.12 |
| 20260818 | 77.64 | 83.54 | **+5.91** | 0.01 | −0.14 |
| 20260819 | 79.95 | 77.80 | −2.15 | −0.06 | −0.01 |

| Metric | Baseline (mean ± SD) | Defended (mean ± SD) |
|---|---|---|
| MAE | **78.90 ± 0.83** | **79.71 ± 4.02** |
| RMSE | 94.73 ± 3.35 | 95.48 ± 3.63 |
| R² | −0.05 ± 0.07 | −0.06 ± 0.08 |

Clean MAE difference (defended − baseline): **+0.81 ± 4.58 K**, sign flipping
across seeds (−4.71, −0.03, +5.05, +5.91, −2.15).

**There is no clean-performance benefit.** The difference is centred near zero
with a standard deviation five times its magnitude. Note also that adversarial
training *increased* run-to-run variance substantially (SD 0.83 → 4.02 K).

### Adversarial validation results

Per-family, mean ± SD of the per-seed statistic across 5 seeds:

| Family | Metric | Baseline | Defended | Mean reduction | Same sign in all 5 seeds? |
|---|---|---|---|---|---|
| Substitution | mean | 9.96 ± 2.33 | 9.14 ± 2.56 | 0.82 | No |
| | P95 | 34.02 ± 16.01 | 32.31 ± 14.26 | 1.70 | No |
| | max | 77.76 ± 34.44 | 60.64 ± 20.28 | 17.12 | No |
| | >52.02 K | 2.1% ± 3.1 | 1.9% ± 2.6 | 0.2 pp | No |
| Insertion | mean | 25.20 ± 4.87 | 23.58 ± 2.83 | 1.61 | No |
| | P95 | 69.69 ± 20.24 | 64.54 ± 8.40 | 5.15 | No |
| | **max** | 144.40 ± 51.70 | 97.65 ± 19.84 | **46.75** | **Yes** |
| | >52.02 K | 10.7% ± 5.2 | 11.0% ± 3.5 | −0.3 pp | No |
| Deletion | mean | 24.54 ± 5.39 | 21.76 ± 2.04 | 2.79 | No |
| | median | 16.49 ± 3.54 | 17.15 ± 2.66 | −0.66 | No |
| | P95 | 68.43 ± 16.83 | 62.14 ± 8.89 | 6.29 | No |
| | max | 136.50 ± 39.82 | 87.87 ± 10.95 | 48.62 | No |
| | >52.02 K | 11.6% ± 5.8 | 8.7% ± 2.3 | 2.9 pp | No |
| Rearrangement | mean | 7.35 ± 1.71 | 6.98 ± 0.84 | 0.37 | No |
| | P95 | 24.22 ± 5.99 | 22.16 ± 2.35 | 2.07 | No |
| | max | 36.90 ± 19.50 | 35.94 ± 5.41 | 0.96 | No |

**Pooled across all seeds (1,767 matched candidate pairs):**

| Quantity | Value |
|---|---|
| Mean \|drift\| baseline | 17.76 K |
| Mean \|drift\| defended | 16.28 K |
| Mean reduction | **1.48 K** |
| Candidates improved | **52.4%** |
| Wilcoxon signed-rank | W=737,380, p=0.042 |

### Interpretation — Phase 2B does NOT replicate

**The large Phase 2B robustness reductions were substantially seed-specific.**

Phase 2B (seed 20260815) reported insertion mean drift 32.5 → 20.7 K (−11.8 K)
and deletion 30.9 → 18.4 K (−12.4 K). Across 5 seeds those same reductions are
**1.61 ± 5.77 K** and **2.79 ± 6.74 K** — an order of magnitude smaller, with
standard deviations exceeding the effects. Seed 20260815 was a favourable draw.

Direction is not consistent either. Of 16 family×metric combinations, only
**one** — insertion max drift — reduced in all 5 seeds. Deletion *median* drift
went the wrong way on average (−0.66 K, i.e. defended was worse).

The pooled effect (1.48 K, 52.4% of candidates improved) is barely above the
50% coin-flip line. The Wilcoxon p=0.042 is reported, but I do **not** claim
statistical significance for the intervention: the test pairs *candidates*, and
candidates within a seed share a model, so the observations are not independent.
The correct unit of independence is the seed (n=5), and at that level the
effects are indistinguishable from noise. A p-value computed over 1,767
non-independent points overstates the evidence.

**What survives:** a consistent reduction in *worst-case* drift for
length-changing attacks — insertion max 144.4 → 97.7 K (all 5 seeds), deletion
max 136.5 → 87.9 K (4 of 5). Defended models are also markedly less variable in
their adversarial behaviour (e.g. insertion max SD 51.70 → 19.84). Adversarial
training appears to clip the tail without shifting the central tendency.

**What does not survive:** any claim about mean or median drift reduction, the
>52.02 K breach rate, or clean performance.

### Statistical honesty

No significance claim is made for the seed-level comparison. With n=5 a t-test
would be underpowered and its normality assumption unverifiable; nothing here
justifies one. The Wilcoxon over matched candidates is reported as a descriptive
paired statistic with its non-independence caveat recorded in
`results/phase2c_audit.json`. Directional consistency across seeds
(`reduction_all_seeds_positive`) is used as the primary robustness criterion,
because with 5 points a sign test is more defensible than a distributional one.

### Test repair (also requested this phase)

Six stale tests were repaired, none deleted. All 200 tests now pass.

| Test | Was | Now |
|---|---|---|
| `test_classify_token` | `TokenRole.ATOM` | Split into `ALIPHATIC_ATOM`/`AROMATIC_ATOM`, `DISCONNECT` separated; new regressions pin both Phase 1D fixes |
| `test_reordering_*` | Tested dead `reordering.py` | Retargeted to `rearrangement.py`, the module all experiments actually use |
| `test_*_stubs_raise` | Asserted `PendingImplementation` | Assert the implementations exist; new test pins the Phase 2B `scaler_path`/`write_back_config`/`seed` controls |
| `test_importing_package_does_not_pull_in_torch` | Whole package torch-free | Scoped to the analysis layers (tokenizer/attacks/validation/evaluation), where the property still holds and matters |
| `test_shipped_configs_..._null` | Asserted configs null | Inverted: asserts configs are resolved, so a clobbered config is caught |
| `test_real_data_dir_is_currently_empty` | Asserted `data/raw` empty | Inverted: asserts the OpenPoly file is present |

**Bug found during repair:** the Phase 1D/1E rewrites of insertion, deletion and
rearrangement **dropped their `@register_attack` decorators**. Only
`substitution` was registered, so `build_attack("deletion")` raised `KeyError`.
Experiments construct attacks directly, so no result is affected — but the
pluggable-attack contract was silently broken. Decorators restored; two tests now
pin it.

Two genuine inconsistencies noted rather than papered over: `RearrangementAttack`
lost its `window_size >= 2` validation (degrades to zero candidates instead of
raising), and substitution guards a missing pool with `PendingImplementation`
while insertion uses a required argument. Both recorded in Problems #13.

### Limitations

1. **5 seeds is a small sample.** SDs are themselves imprecisely estimated. The
   qualitative conclusion (large 2B effects do not replicate) is robust; the
   precise magnitudes are not.
2. **One data split.** All seeds share the same scaffold split, so split-induced
   variance is not captured and is confounded with the small validation set.
3. **Validation n=37.** Clean metrics on 37 samples are noisy; the ±4.02 K
   defended SD partly reflects that.
4. **Negative R² throughout**, as in every prior phase. The model remains a
   controlled attack target, not a competitive predictor.
5. **Adversarial training set held fixed** across seeds. Regenerating it per
   seed would add augmentation variance to the estimate — deliberately excluded,
   so these SDs understate total pipeline variance.
6. **The target-preservation assumption is inherited unchanged** and remains
   chemically unjustified.
7. **Attack candidates differ across seeds by design.** Between-seed drift
   variation therefore mixes model variance with candidate-set variance; only
   within-seed comparisons are matched.

---

## 12. Open Research Questions

1. **How many usable Tg records exist after dedup?** Determines whether a
   Transformer is defensible at all, and at what capacity.
2. **What are the Tg units, and are they mixed?** A K/°C mix would be a ~273-unit
   label error on an unknown subset, invisible except as mediocre RMSE.
3. **How should conflicting duplicates be resolved?** Mean / keep-first / drop are
   different scientific choices. If the cause is a unit mix, averaging manufactures
   a value wrong in **both** units.
4. **What drift threshold makes an attack "successful"?** Should be calibrated to
   baseline test MAE — drift below model error is not obviously meaningful.
5. **Does the validity filter dominate the results?** See §9. Needs drift reported
   conditioned on validity.
6. **Is scaffold splitting appropriate for polymer repeat units?** Bemis–Murcko was
   designed for small molecules; its behaviour on star-containing PSMILES needs
   checking once data exists.
7. **Are heuristic plausibility flags too permissive or too strict?** Currently
   unmeasurable without real data.

---

## 12. Literature

- **OpenPoly** — Wang et al., *Chinese Journal of Polymer Science* (2025),
  DOI `10.1007/s10118-025-3402-y`. Primary dataset. Reports XGBoost > deep
  learning on Tg under data scarcity. **Paywalled; only abstract-level details
  verified.**
- Related polymer benchmarks noted for possible external validation (§18 of
  brief, not Phase 1): PI1M, PolyInfo, Khazana, RadonPy, PolyMetriX, Open
  Polymer Challenge / OPoly26. **None inspected yet.**

---

## 13. Reproducibility

Verified environment (2026-08-15):

```
python   3.12.3        platform Linux-6.18.7-76061807-generic-x86_64-glibc2.39
numpy    2.5.2         pandas   3.0.5      pyyaml 6.0.3
rdkit    2026.03.5     pytest   9.1.1      torch  NOT INSTALLED (deferred)
hardware 12-core i5-12450H, 15Gi RAM (~3.5Gi free), NO GPU
```

Setup and test:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[chem,dev]"
./scripts/run_tests.sh -q      # wrapper needed; see Problem #6
```

Practices in force: configs over hardcoded values (`null` marks unmade
decisions, and `require_resolved()` raises on them); explicit seeds with
`make_rng(seed)` preferred over global state; determinism verified via
**subprocess** (in-process checks hide global-state bugs); splits persisted as a
frozen artifact rather than re-shuffled at runtime; deterministic `attack_id`s;
rejected candidates logged with reasons; import-hygiene test asserts torch never
enters `sys.modules`.

---

## 15. Decision History

*Append-only. Never rewrite.*

**2026-08-15**

1. Confirmed `bio-cyber-adversarial` is unrelated.
2. Adopted persisted split column.
3. Identified OpenPoly.
4. User decision: install RDKit only, defer torch.
5. User decision: strict UNCHECKED validity semantics.
6. Chose signed drift storage.
7. Chose token-level attacks.
8. Made substitution's pool injected.
9. Included Phase 2+ record fields.
10. Replaced unit-check with shift-correlation.
11. Fixed pandas 3.0 string type issue.
12. **Dataset fetched:** 443 Tg records verified.
13. **Conflict Resolution:** Applied `drop_all` to 31 conflicting canonical representations, resulting in 247 usable unique samples.
14. **Baseline Sanity Gate Failed:** Transformer MAE ~319 diagnosed as lack of target normalization, causing model collapse to ~143 K.
15. **Target Normalization:** Implemented `TargetScaler` fitted exclusively on training set to resolve collapse.
16. **Baseline Sanity Gate Passed:** Experiment 1B completed successfully with Test MAE 52.02 (vs trivial 132.54) and prediction sensitivity confirmed.
17. **Phase 1D Preliminary Smoke Test:** "Unconstrained Token-Level Substitution Baseline" discovered that `= -> .` was incorrectly admitted by the initial role-preserving taxonomy.
18. **Phase 1D Refined Executed:** Token taxonomy tightened. Aliphatic and aromatic atoms strictly separated. Disconnect (`.`) separated from bond tokens. Valid candidate rate increased to 52.2%.
19. **Phase 1E Designed:** Defined Insertion, Deletion, and Local Rearrangement attacks. Implemented unit tests guaranteeing protected-token constraints (e.g. `(`, `[*]`).
20. **Phase 1E Executed:** Ran comparative sequence perturbation study. Discovered Insertion and Deletion drive significantly higher median and max drift than Substitution and Rearrangement.
21. **Phase 1F Executed:** Conducted perturbation attribution analysis. Concluded that the massive prediction drift is empirically driven by sequence-length changes (`delta_length != 0`), not by the number of edited tokens.
22. **Phase 2A Designed & Executed:** Augmented the training dataset with a 25/25/25/25 mixture of the four attack classes under a target-preservation assumption. Clean validation MAE improved (52.02 K -> 46.20 K) and adversarial drift was massively reduced (Insertion mean drift -11K, Deletion max drift -73K).
23. **Phase 2A Evaluation Audit:** Programmatically verified train/test isolation, 0 overlap between validation and train adversarial strings, and confirmed the exact validation drift metrics. Identified a subtle scaler-fitting confounding factor.
24. **Phase 2B Designed & Executed (Controlled Defense Ablation):** Re-ran adversarial training with the Phase 1 `TargetScaler` **loaded rather than refit**, holding normalization statistics fixed (mean 330.713124, std 88.046275). Added `scaler_path` and `write_back_config` to `train()`.
25. **Phase 2B — Second duplicate class found:** Beyond the 11 adversarial-vs-adversarial duplicates reported in 2A, **19** adversarial strings were byte-identical to clean training strings — pure sample-weight inflation. 30 duplicates removed in total, leaving 1,082 adversarial examples.
26. **Phase 2B — Balancing rejected as primary, run as diagnostic:** Equalising attack families would discard 54.2% of adversarial data (rearrangement is validity-limited at 124), introducing a second confounder. The balanced diagnostic scored 68.56 K test MAE, materially worse, confirming augmentation volume dominates family balance at this data scale.
27. **Phase 2B — Config write-back bug found:** the Phase 2A run had overwritten `configs/model.yaml:baseline_metrics` with the *defended* model's numbers, so the attack threshold was silently reading 46.20 rather than 52.02. Runs now write `metrics.json` beside their own checkpoint, and the threshold is passed explicitly.
28. **Phase 2B Result — clean gain does NOT survive the control:** Test MAE 52.02 -> **50.64 K** with the scaler fixed, versus 46.20 K in 2A. Only **1.38 K of the 5.81 K** apparent improvement survives (~24%); **4.44 K is associated with the normalization change**. The Phase 2A clean-performance claim is therefore not robust.
29. **Phase 2B Result — robustness DOES survive the control:** Under a fixed scaler, a fresh paired attack set (seed 20260902, 22.0% overlap with Phase 1E) and identical candidates for both models, pooled mean |drift| fell 21.64 -> 13.16 K (-8.48 K, 63.9% of candidates improved), with per-family reductions across substitution, insertion and deletion. This is the defensible Phase 2 finding. *(SUPERSEDED by Phase 2C — see #32. This was a single seed and does not replicate.)*
30. **Phase 2C Designed & Executed:** 5 seeds x 2 conditions = 10 trainings under the exact 2B controls, with the Phase 1 scaler frozen in all 10 (verified identical). Added a `seed` argument to `train()`. Test set never loaded; all reporting on validation (n=37).
31. **Phase 2C — stale tests repaired, registry bug found:** All 6 obsolete tests updated to the current contract (none deleted); 200 pass. Discovered the Phase 1D/1E rewrites had dropped `@register_attack` from insertion, deletion and rearrangement, leaving only substitution registered. No results affected (experiments construct attacks directly); decorators restored.
32. **Phase 2C Result — Phase 2B does NOT replicate.** Seed 20260815 gave insertion/deletion mean-drift reductions of ~12 K; across 5 seeds those are **1.61 ± 5.77 K** and **2.79 ± 6.74 K**, SDs exceeding the effects. Only **1 of 16** family x metric combinations (insertion max drift) reduced in all 5 seeds. Pooled reduction 1.48 K, 52.4% of candidates improved — barely above chance. **Phase 2B landed on a favourable seed.**
33. **Phase 2C Result — no clean benefit:** defended − baseline validation MAE is **+0.81 ± 4.58 K**, sign-flipping across seeds. Adversarial training also *increased* run-to-run variance (SD 0.83 -> 4.02 K).
34. **Phase 2C — what survives:** worst-case drift reduction for length-changing attacks (insertion max 144.4 -> 97.7 K in all 5 seeds; deletion max 136.5 -> 87.9 K in 4 of 5) and markedly lower variance in adversarial behaviour. Adversarial training clips the tail without shifting central tendency. No significance claim is made at the seed level (n=5).
35. **Phase 2F Forensic Audit — Phase 1F REFUTED.** Length-preserving controls (token shuffle, sequence reverse) produce drift of 32.05 K and 30.44 K versus 24.96 K for a length-increasing edit, consistently across all 5 seeds. The model is globally unstable to token perturbation, not specifically length-sensitive. Phase 1E's "length-preserving" families were mis-specified (rearrangement is an isomerization with ΔMW = 0.00 and 100% formula preservation).
36. **Phase 2F — length shortcut found.** A length-only linear regression scores 76.77 K validation MAE versus the Transformer's 79.04 K. corr(length, Tg) = 0.372 across 247 samples; the scaffold split sorted long/high-Tg polymers into val/test (mean Tg 330.7/384.5/463.2 K). Bag-of-tokens ridge ties the Transformer. The claim that the model learned chemistry is withdrawn.
37. **Phase 2F — masking verified correct.** PAD-masked padding is an exact no-op (0.0000 K at +100 pads); predictions are batch- and width-invariant to 3e-05. Masking, pooling and evaluation bugs are refuted. Mean-pool dilution confirmed instead: drift ∝ 1/L, corr(1/L, insertion drift) = +0.575.
38. **Phase 2F — target-preservation hole widened.** No attack family is representation-level; substitution/insertion/deletion alter molecular formula and rearrangement is an isomerization. 12.9% of "insertion" candidates are canonically identical to the original. SMILES randomization identified as the missing provably-label-preserving control.

---

---

## Phase 2F: Forensic Architecture Audit

*Executed 2026-08-15. All prior records preserved unchanged. Full detail in
[ARCHITECTURE_FORENSIC.md](ARCHITECTURE_FORENSIC.md); reproduce with
`.venv/bin/python scripts/forensic_diagnostics.py`.*

An adversarial audit whose goal was to falsify, not support, the existing
conclusions. It succeeded on the central one.

### Headline: Phase 1F is REFUTED

Phase 1F concluded that length-changing perturbations cause the ~30 K drift.
A **length-preserving** control — shuffling or reversing the token sequence,
which preserves the token multiset and the molecular formula exactly — produces
drift **of the same or larger magnitude, in all 5 seeds**:

| Seed | shuffle (len-preserving) | reverse (len-preserving) | duplicate token (len +1) |
|---|---|---|---|
| 20260815 | 36.03 | 38.58 | 38.66 |
| 20260816 | 40.47 | 37.13 | 24.92 |
| 20260817 | 20.07 | 17.61 | 18.80 |
| 20260818 | 24.60 | 23.35 | 18.68 |
| 20260819 | 39.07 | 35.51 | 23.72 |
| **mean** | **32.05** | **30.44** | **24.96** |

The model is **globally unstable to token-level perturbation**, not specifically
length-sensitive. Phase 1E's "length-preserving" families were mis-specified:
rearrangement permutes a 3-token window (ΔMW = 0.00, formula preserved in 100%
of cases — an isomerization), and substitution was role-constrained. Neither was
a strong order perturbation.

### The model may not have learned chemistry

Control baselines (MAE, K; test n=6 is anecdote):

| Model | Val MAE | Test MAE |
|---|---|---|
| Mean predictor | 88.48 | 132.54 |
| **Length-only linear regression** | **76.77** | 76.39 |
| Bag-of-tokens ridge (α=1) | 79.59 | 43.24 |
| **Transformer (Phase 1)** | **79.04** | 52.02 |

A one-parameter model on sequence length beats the Transformer on validation.
corr(length, Tg) = 0.372 over all 247 samples. The scaffold split is confounded:
train/val/test mean Tg = 330.7 / 384.5 / 463.2 K and mean length = 22.6 / 43.5 /
56.7. Test Tg lies entirely above the training mean, which explains the
universally negative R².

### What is definitely correct

Padding, masking, batching and the scaler are **verified correct**. Appending
PAD-masked positions is an exact no-op (max drift 0.0000 K at +100 pads);
predictions are batch- and padding-width-invariant to 3e-05. Hypotheses of a
masking bug, pooling bug or evaluation bug are **refuted**.

### Mechanisms identified

- **Mean-pool dilution:** drift scales as 1/L (corr(1/L, insertion drift) =
  +0.575). Short sequences drift 75 K, long ones 19 K. Per-sample drift is
  **not comparable across lengths**, and val/test are systematically longer.
- **Positional encoding contributes ~5 K of ~30 K** — real but minor (measured
  by appending unmasked PAD tokens: pure position shift, no chemistry).
- **Attack families are not severity-matched:** mean |ΔMW| is 0.00
  (rearrangement) vs 15.75 (insertion). Phase 1E confounds mechanism with
  chemical severity. 12.9% of "insertion" attacks yield a molecule identical to
  the original after canonicalization.

### Target-preservation assumption — worse than recorded

No current attack family is a legitimate representation-level perturbation.
Substitution/insertion/deletion change molecular formula; rearrangement is an
isomerization. The label-preserving assumption is chemically unjustified for
**every** family. The missing control is **SMILES randomization** — different
valid writings of the identical molecule, where Tg is provably unchanged.

### Claims now withdrawn

1. ❌ "Length-changing perturbations cause the vulnerability" — refuted.
2. ❌ "Positional encoding is the mechanism" — ~5 K of ~30 K.
3. ❌ "The Transformer learned polymer chemistry" — length-only beats it.
4. ❌ "Adversarial training improves chemical invariance" — no family preserves
   chemistry.

Phase 2C's conclusion (2B does not replicate) **stands and is reinforced**.

---

## 16. NEXT ACTION

> **Phase 3 remains on hold. Phase 2F has overturned the Phase 1F mechanism.**
>
> Recommended order:
>
> 1. **SMILES-randomization invariance test** — the only provably
>    Tg-preserving perturbation. No training required. Converts "drift" from an
>    assumption-laden quantity into unambiguous model error.
> 2. **Positional-encoding ablation** (5 seeds, no positional embedding) — makes
>    the mechanism claim causal rather than observational.
> 3. **k-fold scaffold cross-validation** of length-only / bag-of-tokens /
>    Transformer — replaces the n=6 test set and settles whether the Transformer
>    is learning anything beyond length and token counts.
>
> Do **not** add attack families, retrain the defended model, or start
> multi-step/MCMC attacks until the taxonomy in §6 of the forensic doc is fixed.

---

## 16b. Superseded NEXT ACTION (pre-Phase-2F, retained)

> **Review Phase 2C. Phase 3 is NOT started and remains on hold.**
>
> The multi-seed confirmation is complete and it overturns the Phase 2B headline.
>
> - **Phase 2B's large robustness effects do NOT replicate.** Its ~12 K
>   insertion/deletion mean-drift reductions become 1.6 ± 5.8 K and 2.8 ± 6.7 K
>   across 5 seeds. Only 1 of 16 family×metric combinations reduced in every
>   seed. Phase 2B landed on a favourable seed.
> - **No clean-performance benefit exists.** Defended − baseline validation MAE
>   is +0.81 ± 4.58 K, sign-flipping across seeds.
> - **What survives:** worst-case (max) drift reduction for length-changing
>   attacks, consistent across seeds, plus markedly lower variance in adversarial
>   behaviour. Adversarial training clips the tail; it does not shift the centre.
>
> The defensible claim has narrowed from "adversarial training reduces prediction
> drift" to **"adversarial training reduces worst-case drift under length-changing
> perturbations, without improving average-case drift or clean accuracy."**
>
> Decisions needed before any Phase 3 work:
>
> 1. **Accept the narrowed claim, or investigate further?** The tail-clipping
>    effect is real and consistent but much weaker than 2A/2B suggested.
> 2. **Address the n=37 validation / n=6 test problem.** With 247 total records
>    the split is the binding constraint on every conclusion. Cross-validation
>    over multiple splits would separate model variance from split variance —
>    currently confounded.
> 3. **Reconsider whether this target model is worth defending.** R² is negative
>    for every model in every phase. Robustness of a model that underperforms the
>    mean predictor is of limited scientific interest, and a reviewer will say so.
>
> Recommended next step: **multi-split cross-validation** before Phase 3. Phase 2C
> showed seed variance dominates; split variance is likely larger still and remains
> entirely unmeasured. Building multi-step attacks on top of an effect this small
> would compound an unresolved uncertainty rather than resolve it.
