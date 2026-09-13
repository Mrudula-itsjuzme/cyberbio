# Project Evolution and Model Reasoning

This document serves as the canonical narrative of the materials-adversarial project. It traces the project's evolution, the research questions asked, the hypotheses tested, the failed approaches, and how the final robustness pipeline emerged.

---

## SECTION 1 — PROJECT GOAL

The active project focuses on **polymer sequence modelling**. Our goal is to predict the **Bandgap (in eV)** of polymers using a Transformer-based regressor trained on the **polyVERSE dataset** (4,209 usable records).

### The Central Problem
The core challenge is adversarial robustness in molecular sequence models. The same scientific model can be sensitive to two distinct classes of perturbations:

1. **Representation changes that do NOT change the underlying molecule.** (e.g., randomized SMILES syntax representing the exact same molecular graph).
2. **Chemistry-changing edits that DO change the molecule.** (e.g., substitution, insertion, deletion of monomers).

These must be treated differently because representation changes should ideally result in *identical* predictions (since the physical molecule hasn't changed), while chemistry-changing edits *should* result in different predictions (since the molecule is different). A robust model must be invariant to the former but appropriately sensitive (but not erratically fragile) to the latter.

---

## SECTION 2 — DATA AND SPLITS

* **Dataset Name:** polyVERSE
* **Usable Record Count:** 4,209
* **Target Property:** Bandgap
* **Units:** eV
* **Train / Validation / Exposed-Test Count:** 80% / 10% / 10% splits
* **Split Seed:** 42 (Frozen)
* **Tokenizer/Vocab:** Character-level tokenizer fit on the training set
* **Target Scaler:** StandardScaler fit exclusively on the clean training set

**Why validation is used for model selection:**
We use the validation split to tune hyperparameters, select the best epochs, and evaluate architectural modifications (like robustness losses).

**Why the test split is considered exposed/non-pristine:**
The test split was used during earlier phases of the project before strict data-hygiene boundaries were fully formalized. Because early model iterations and hyperparameter decisions were influenced by test split performance, it is no longer a pristine holdout set.

**Why test metrics are only reference metrics now:**
Due to the aforementioned exposure, test metrics cannot serve as an unbiased measure of true generalization. They are kept purely as reference metrics, while our strict robustness evaluations are performed exclusively on frozen candidate banks derived from the validation split.

---

## SECTION 3 — BASELINE MODEL

The ordinary Transformer architecture is a sequence-to-sequence model adapted for regression.

**Architecture Flow:**
`Input PSMILES` → `tokenization` → `embeddings` → `positional embeddings` → `2 Transformer encoder layers` → `4 internal self-attention heads` → `masked mean pooling` → `regression output`

**What the 4 attention heads actually do:**
They are standard self-attention heads. They are **NOT** assigned to attack families. Self-attention in this project means the model learns relationships among sequence tokens and positions that help Bandgap prediction. It does *not* mean the heads isolate specific chemical or adversarial concepts unless explicitly trained to do so.

**Model Specifications:**
* `d_model` = 64
* `layers` = 2
* `internal attention heads` = 4
* `feedforward size` = 128
* `dropout` = 0.1
* `parameter count` = 85,761

**Baseline Metrics:**
* Clean Validation MAE: ~0.460 eV

---

## SECTION 4 — ATTACK SEMANTICS

| Perturbation Type | Identity Preserved? | Inherit Measured Label? | Used for Training? | Used for Eval? | Current Status |
| --- | --- | --- | --- | --- | --- |
| Randomized SMILES | Yes | Yes (Canonical Equivalence) | Yes (Phase 4) | Yes | ACTIVE |
| Substitution | No | No | Yes (Phase 4, Teacher Distillation) | Yes | ACTIVE |
| Deletion | No | No | No | Planned | FUTURE (Phase 5) |
| Insertion | No | No | No | No | PARKED |
| Rearrangement | No | No | No | No | PARKED |
| Metropolis-style search | No | No | No | No | PARKED |

**Important Distinctions:**
* **Randomized SMILES:** Representation-preserving. Canonical equivalence is required. The measured Bandgap label *may* be reused for training and evaluation.
* **Substitution/Insertion/Deletion/Rearrangement:** Chemistry-changing. The measured label may **NOT** be inherited because the molecule's true property is unknown. These are evaluated as *prediction sensitivity/stress* unless the true property is independently available.

---

## SECTION 5 — WHY WE TRIED SPECIALIZED BRANCHES

**The Hypothesis:**
Could the model learn to separate two robustness concepts internally?
* **Branch A:** Learn representation invariance (randomization).
* **Branch B:** Learn chemistry-change sensitivity (substitutions).

These were **NOT** standard Transformer attention heads. They were explicit learned linear branches placed on top of a shared Transformer encoder.

**Architecture Flow:**
`shared pooled embedding` → `Branch A` & `Branch B` → `fusion` → `Bandgap regressor`

**Why an architecture-control model was necessary:**
To ensure that any improvements were due to the *robustness objectives* and not simply because the model had more parameters or a different structure, we trained an identical two-branch model using only standard MSE loss on clean data (the Architecture Control).

---

## SECTION 6 — SPECIALIZATION LOSS DESIGN

To force the branches to specialize in Phase 3, we designed explicit loss terms:

1. **Property loss (L_prop):** Clean Bandgap prediction.
   * *Equation:* `MSE(fusion(emb), target)`
2. **Representation-invariance loss (L_inv):** Equivalent randomized SMILES should map close in Branch A.
   * *Equation:* `MSE(BranchA(emb_clean), BranchA(emb_rand))`
3. **Chemistry-sensitivity loss (L_sens):** Substitution pairs should create controlled separation in Branch B.
   * *Equation:* `max(0, margin - ||BranchB(emb_clean) - BranchB(emb_sub)||)`
4. **Diversity loss (L_div):** Prevent both branches from becoming identical.
   * *Equation:* `CosineSimilarity(BranchA, BranchB)` penalty.

**Intended Learning:**
The property loss anchors the model to reality. The invariance loss teaches Branch A to ignore syntax changes. The sensitivity loss teaches Branch B to recognize when the chemistry has actually changed. The diversity loss ensures the branches don't just collapse into doing the exact same thing.

---

## SECTION 7 — WHAT ACTUALLY HAPPENED

**The Phase 3 Result: The branches differentiated numerically, but semantic specialization was NOT demonstrated.**

The frozen evaluation showed:
* Branch A became low-response / low-variance (variance collapsed to ~0.036).
* Branch B became high-response / high-variance.
* Selectivity ratios (how much a branch responded to its assigned attack vs the other attack) were similar to the architecture control.
* Specialized losses did not produce unique semantic behavior.
* Substitution robustness was not improved.

**Representation Collapse/Polarization:**
The model found an easier optimization shortcut. Instead of learning complex chemical semantics, it minimized the losses by:
* **Branch A:** Reducing its overall variance and responding weakly to *everything* (making it easy to minimize `L_inv`).
* **Branch B:** Carrying almost the entire predictive and sensitivity burden to satisfy `L_prop` and `L_sens`.

This satisfies the mathematical losses partially without learning the intended human-interpretable semantics.

**Scientific Verdict:** SEMANTIC SPECIALIZATION NOT SUPPORTED.
This is not a project failure; it is a falsified hypothesis. Forcing internal representation boundaries on an ambiguous latent space leads to optimization pathology, not semantic disentanglement.

---

## SECTION 8 — WHY WE CHANGED DIRECTION

**The Decision:**
Instead of forcing internal latent semantics, we decided to move robustness supervision to the **OUTPUT** level.

**Research Question:**
Can the model learn stable predictions directly, without requiring internal branch roles?

This approach is simpler, more direct, and highly falsifiable. Rather than telling the model *how* to represent concepts internally, we simply penalize it when its final predictions behave erratically.

---

## SECTION 9 — OUTPUT-LEVEL ROBUSTNESS TRAINING

In Phase 4, we used two output-level objectives:

### A. Randomization consistency
For an input `x` and its randomization `x_rand` (same molecule), we train the model so predictions remain stable.
* Measured target allowed on both.
* Added a prediction-consistency loss: `lambda_consistency * MSE(pred_clean, pred_rand)`.

### B. Substitution teacher consistency
For an input `x_sub` (chemistry has changed), we do **NOT** use the original measured Bandgap, because the target property is unknown.
Instead, we use a frozen teacher model's prediction (the Architecture Control model) as a distillation target.
* Added a teacher-consistency loss: `lambda_teacher * MSE(pred_sub, teacher_target_sub)`.

**Important Disclaimer:** This teacher consistency is *not* a claim about the true physical Bandgap of the substituted molecule. It is a distillation/stability target designed to prevent the robust model from failing wildly on out-of-distribution syntax.

---

## SECTION 10 — MODEL COMPARISON LOGIC

We evaluate four relevant model conditions to isolate different effects:

1. **Ordinary Transformer baseline**
2. **Two-branch architecture control**
3. **Randomization-robust model**
4. **Mixed-robust model**

**Isolation Logic:**
* `Baseline` vs `Architecture Control` → Isolates the **architecture effect**.
* `Architecture Control` vs `Randomization Robust` → Isolates the **randomization consistency-training effect**.
* `Randomization Robust` vs `Mixed Robust` → Isolates the added **substitution-teacher distillation effect**.

---

## SECTION 11 — PHASE 4 RESULTS

**Clean Validation MAE:**
* Ordinary Baseline: 0.460 eV
* Architecture Control: 0.461 eV
* Randomization-Robust: 0.443 eV
* Mixed-Robust: 0.441 eV

**Randomization Drift:**
* Architecture Control: 0.382 eV
* Randomization-Robust: 0.301 eV
* Mixed-Robust: 0.308 eV

**Substitution Stress Drift:**
* Architecture Control: 0.281 eV
* Mixed-Robust: 0.269 eV

**Current Strongest Findings:**
* **Randomization consistency training** provides a substantial reduction in randomization drift beyond the architecture control, and it simultaneously improved clean validation performance.
* **Mixed training** provided a modest improvement in substitution stability, retained good clean performance, and largely preserved the randomization robustness gains.

*Note: These results provide evidence that output-level consistency training is effective, but they do not justify exaggerated claims of perfect robustness.*

---

## SECTION 12 — HOW THE MODEL "THINKS"

Do NOT anthropomorphize the model. The model computes mathematically:

`Input tokens` → `contextual token representations` → `pooled molecular-sequence representation` → `branch transformations` → `fused representation` → `scalar Bandgap prediction`.

**What robustness training changes:**
It does not teach the model human chemical reasoning. It changes the optimization pressure so that:
* Equivalent strings are encouraged to produce similar predictions.
* Chemistry-changing perturbed inputs are constrained by teacher behavior to avoid erratic extrapolations.
* The model becomes less sensitive to superficial representation choices.

We do **NOT** know the explicit internal reasoning of the model, and we do **NOT** claim attention heads learned chemistry concepts, as this was not experimentally shown.

---

## SECTION 13 — EVERY MAJOR QUESTION ASKED SO FAR

* **Q1: Can a Transformer predict polymer Bandgap reasonably well?**
  * **Why we asked:** To establish a baseline.
  * **Experiment:** Train a standard Transformer regressor.
  * **Result:** Achieved ~0.460 eV MAE.
  * **Decision:** Proceed to adversarial testing.

* **Q2: Is it sensitive to representation perturbations?**
  * **Why we asked:** To verify if the model is robust to syntax changes.
  * **Experiment:** Evaluated Randomization and Substitution attacks.
  * **Result:** High drift (e.g., 0.614 eV on randomization).
  * **Decision:** The model is highly sensitive; defense is needed.

* **Q3: Can we assign internal branches to different robustness concepts?**
  * **Why we asked:** Hypothesis that semantic specialization helps robustness.
  * **Experiment:** Built Two-Branch architecture with specialized losses (Phase 3).
  * **Result:** Branches polarized in variance, but didn't separate concepts.
  * **Decision:** Reject hypothesis, seek alternative.

* **Q4: Do those branches genuinely specialize?**
  * **Why we asked:** To verify Q3.
  * **Experiment:** Analyzed selectivity ratios of Branch A vs Branch B.
  * **Result:** No semantic specialization observed.
  * **Decision:** Abandon internal latent specialization.

* **Q5: Does architecture alone improve robustness?**
  * **Why we asked:** To control for increased parameter count in the two-branch model.
  * **Experiment:** Trained Two-Branch Architecture Control on clean data.
  * **Result:** Architecture alone reduced randomization drift (0.614 -> 0.382 eV).
  * **Decision:** Use Architecture Control as the strict baseline for future defenses.

* **Q6: Does output-level randomization consistency improve robustness beyond architecture?**
  * **Why we asked:** To test if output-level regularization works better than internal specialization.
  * **Experiment:** Phase 4 Randomization-Robust training.
  * **Result:** Significant improvement in randomization drift (0.382 -> 0.301 eV) and clean MAE (0.461 -> 0.443 eV).
  * **Decision:** Adopt this training objective.

* **Q7: Does substitution teacher-consistency provide an additional benefit?**
  * **Why we asked:** To defend against chemistry-changing adversarial edits.
  * **Experiment:** Phase 4 Mixed-Robust training with a frozen teacher.
  * **Result:** Modest improvement in substitution drift (0.281 -> 0.269 eV) while maintaining randomization defense and clean MAE.
  * **Decision:** Adopt as the final robustness pipeline.

* **Q8: Does robustness transfer to an unseen chemistry-changing attack?**
  * **Status:** NOT YET TESTED — Planned for Phase 5 (Deletion).

---

## SECTION 14 — METRICS EXPLAINED

* **MAE (Mean Absolute Error):** The average absolute difference between the predicted and true Bandgap.
* **RMSE (Root Mean Squared Error):** Penalizes larger errors more strongly than MAE.
* **R²:** The fraction of target variance explained relative to a mean-prediction baseline.
* **Prediction drift:** The absolute difference between the prediction on the original representation and the perturbed representation.
* **Stress exceedance:** The fraction of perturbed samples whose drift exceeds a fixed threshold (e.g., the clean-baseline validation MAE).
* **Bootstrap confidence interval:** Uncertainty on the paired robustness improvement, calculated by resampling source polymers to ensure statistical significance.

*Note: "Accuracy" is not the correct term for this regression problem. We use MAE, RMSE, and prediction drift.*

---

## SECTION 15 — WHAT COUNTS AS A GOOD RESULT

Low clean MAE alone is not the only objective. A robust model should ideally have:
* Good clean MAE
* Low representation drift
* Acceptable chemistry-stress behavior
* Improvement beyond architecture controls
* Reproducible paired evaluation

A slightly worse clean model can still be scientifically useful if robustness improves strongly. However, in our current Phase 4 results, clean performance actually improved, which is a highly favorable outcome.

---

## SECTION 16 — FAILED / REJECTED IDEAS

| Idea | Reason tried | Observed problem | Final status |
| --- | --- | --- | --- |
| One-head-per-attack-family interpretation | Attempt to make self-attention explainable | Attention heads do not map neatly to human concepts | REJECTED |
| Semantic branch specialization | Attempt to force disentanglement of robustness concepts | Representation collapse/polarization; no actual specialization | REJECTED |
| Label inheritance for chemistry-changing edits | To calculate robust MAE on substitutions | The true Bandgap changes when the molecule changes | REJECTED |
| Treating all perturbations as equivalent attacks | Simplicity | Conflates representation-invariance with chemistry-sensitivity | REJECTED |
| Overreliance on attention weights for explanations | Interpretability | Attention is not explanation | REJECTED |
| Formal "MCMC" terminology | To describe substitution search | Mathematically unsupported for our simple greedy search | REJECTED |

---

## SECTION 17 — CURRENT PROJECT STATE

**Active Final Pipeline:**
`polymer PSMILES` → `Transformer / two-branch architecture` → `clean Bandgap prediction` → `robustness training` (randomized SMILES consistency + substitution teacher consistency) → `frozen paired validation evaluation`.

**Main attack families kept:**
1. Randomization
2. Substitution

**Upcoming transfer-only attack:**
3. Deletion (Do not expand attack families unless later justified).

---

## SECTION 18 — WHAT IS NEXT

**Planned Phase 5:**
Unseen deletion-transfer evaluation. We will generate deletion candidates and evaluate the frozen models without any retraining.
*Question:* Does robustness learned from randomization + substitution transfer to deletion?

**Future Work (Not in Phase 5):**
* Insertion
* Rearrangement
* Adaptive search
* Metropolis-style search
* LLM proposal policy
* GAN/deepfake/genomic extensions

*Do NOT mix future work with current results.*

---

## SECTION 19 — CLAIMS LEDGER

**SUPPORTED CLAIMS:**
* Equivalent-SMILES consistency training reduces randomization drift.
* Architecture control improves over ordinary baseline in some metrics.
* Teacher-distillation provides modest improvements in substitution stability.
* Output-level robustness training does not inherently degrade clean validation MAE.

**PARTIALLY SUPPORTED CLAIMS:**
* None currently requiring caveat.

**REJECTED CLAIMS:**
* Explicit branch losses create semantic head specialization.
* Internal latent space can be easily forced into human-interpretable orthogonal concepts.

**NOT YET TESTED CLAIMS:**
* Robustness transfers to unseen attacks (e.g., deletion).

---

## SECTION 20 — RESULT PROVENANCE

* **Ordinary Baseline:**
  * Model Checkpoint: `results/models/transformer_regressor/model_epoch_10.pt`
  * Checkpoint Hash: `13fc7a7` (Phase 1)
  * Scaler Hash: `eb783b9`
  * Evaluator: `scripts/evaluate_phase4.py`
* **Architecture Control:**
  * Model Checkpoint: `results/models/specialized_control/model_epoch_07.pt`
  * Checkpoint Hash: `db9cfb5` (Phase 3)
* **Randomization-Robust (lambda_cons=1.0):**
  * Model Checkpoint: `results/phase4_controlled_robustness/rand_robust_1.0/model_epoch_02.pt`
* **Mixed-Robust (lambda_teach=0.1):**
  * Model Checkpoint: `results/phase4_controlled_robustness/mix_robust_0.1/model_epoch_20.pt`
* **Evaluation Banks:** `data/phase3_frozen/`
  * Randomization hash: `23c0429`
  * Substitution hash: `785d775`

---

## SECTION 21 — DOCUMENTATION STYLE

This document is written to be accessible to beginners trying to understand the whole project, while remaining precise enough for a reviewer checking scientific rigor. It emphasizes plain-English explanations first, followed by technical details, and explicitly separates hypotheses from proven results.
