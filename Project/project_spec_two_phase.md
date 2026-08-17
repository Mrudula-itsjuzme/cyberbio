# PROJECT BUILD SPEC — "Learning to Attack and Defend"
### Unified Adversarial Framework for Robust Materials Sequence Modelling

> **How to use this file:** This document is written so that it can be re-uploaded to Claude on its own, in a new conversation, and Claude can generate the full project from it with no further context needed. It fixes the scope, the architecture, the file structure, and the exact deliverables for each phase, so nothing has to be re-explained or re-decided. When re-uploading, just say which phase to build (e.g. "Build Phase 1 from this spec") and Claude should proceed directly to producing code/notebooks, not ask clarifying questions, unless something in this spec is genuinely ambiguous.

---

## 0. Fixed Project Context (do not re-derive, just use this)

- **Title:** Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling
- **Domain:** Adversarial ML applied to materials-property prediction over sequence representations (chemical formulas, SMILES, polymer SMILES, crystal/material sequences).
- **Core idea:** A probabilistic attack generator (MCMC-based) proposes small, scientifically plausible modifications to a material sequence. These are evaluated against a target deep learning property-predictor to measure prediction drift / fragility, then reused as adversarial training data to make the target model more robust. This forms a closed loop: **Generate → Attack → Evaluate → Train → Improve → Attack Again.**
- **Role of probabilistic reasoning:** Confined to the *attack-generation* stage only. It answers "what scientifically plausible modification should we try next?" via a learned transition/mutation model + MCMC sampling (propose → check plausibility → evaluate adversarial effect → accept/reject). It is not a separate module bolted on for its own sake.
- **Role of deep learning:** The target property-prediction model (Transformer / GNN / sequence model depending on representation) that gets attacked and then defended via adversarial training.
- **Constraints carried over from prior related work (`madvac` framework):**
  - No external cheminformatics dependencies (no RDKit-hard-requirement, etc.) — build self-contained tokenization/validity logic, or make cheminformatics libraries optional with a fallback.
  - Should work end-to-end on synthetic/mock data if a real dataset isn't available, so the pipeline is always runnable and demonstrable.
  - Deliver as clean, modular Python — notebook-friendly (Colab-style) for demonstration, but organized enough to reuse as a package.
- **Representations in scope:** chemical formulas, SMILES, polymer SMILES, generic material/crystal sequences — pick the SMILES + chemical formula path as the primary demo path unless told otherwise, since it generalizes most easily to a toy synthetic dataset.
- **Total timeline (from Gantt chart):** 4 weeks / ~30 days, roughly 50% of tasks completed by end of Week 1.

### Full 14-task list (from the Gantt chart), for reference/traceability
1. Literature Review — **DONE**
2. Dataset Collection & Selection
3. Data Preprocessing & Cleaning
4. Sequence Representation (Tokenizer / Encoder)
5. Baseline DL Model (Training)
6. Initial Attack Engine Design
7. Build Probabilistic Model (Transitions / HMM)
8. Implement MCMC / Sampling Engine
9. Scientific Validity & Plausibility Checks
10. Generate Adversarial Sequences
11. Attack Evaluation & Fragility Analysis
12. Adversarial Training (Retraining / Fine-tune)
13. Robustness Testing (Unseen Attacks)
14. Analysis, Visualization & Final Report

---

## 1. PHASE 1 — Foundation + Probabilistic Attack Engine
**Covers Gantt tasks 2–10 (Weeks 1–2)**

### Goal of this phase
Produce a working pipeline that: loads/represents material sequence data → trains a baseline property-prediction model → builds a probabilistic (MCMC) attack generator that produces scientifically plausible adversarial sequences → runs those attacks against the baseline model and shows prediction drift. **No defense/retraining yet — this phase ends once attacks are generated and shown to affect the baseline model.**

### Deliverables (build all of these)
1. **Dataset module** (`data.py` / notebook section)
   - Loads a real small materials dataset if available (e.g. a small public SMILES-property dataset), OR generates a clearly-labeled **synthetic dataset** of chemical-formula/SMILES-like strings with a synthetic target property (regression value), if no real dataset is supplied. Must state clearly in output which mode was used.
   - Cleaning: dedup, filter invalid/malformed strings, train/val/test split.
2. **Sequence representation module** (`representation.py`)
   - Tokenizer for SMILES/formula strings (character-level or rule-based token vocabulary — self-contained, no hard RDKit dependency).
   - Encoding utilities (token → id, padding, batching).
3. **Baseline deep learning model** (`model.py`)
   - A sequence model (start with a small Transformer encoder or BiLSTM — keep it small/fast to train, this is a baseline not a SOTA model) that predicts a material property (regression) from the tokenized sequence.
   - Training loop with loss curve, basic metrics (MAE/RMSE), and a saved/loadable checkpoint.
4. **Probabilistic attack engine** (`attack_engine.py`) — the core of this phase
   - **3.1 Transition/mutation model:** learn or define `P(token_j | token_i)` from the training data (simple learned transition matrix is fine — this doesn't need to be a full HMM unless you want to extend it) to know which token substitutions are statistically/scientifically reasonable.
   - **3.2 MCMC sampler:** implement the loop — start from an original sequence → propose a local modification sampled from the transition model → check plausibility → score adversarial effect (via the baseline model) → accept/reject (Metropolis-style acceptance) → repeat to produce a chain of candidate adversarial sequences.
   - **3.3 Scientific validity / plausibility checker:** rule-based checks (valid token grammar, valency-style constraints if feasible without RDKit, or at minimum sequence-structure validity for the synthetic representation) that reject nonsensical candidates before they reach the model.
5. **Adversarial sequence generation demo**
   - Run the attack engine over a sample of test sequences, output a table: original sequence, adversarial sequence, original prediction, adversarial prediction, prediction drift, whether accepted as plausible.
6. **Phase 1 notebook/report** tying it together with: pipeline diagram recap, sample generated adversarial sequences, a plot of prediction drift distribution, and 2–3 sentence summary of baseline model fragility observed.

### Explicitly out of scope for Phase 1
- Adversarial training / retraining
- Robustness testing on unseen attacks
- Final comparison against random/rule-based baselines (that's Phase 2's evaluation section, though the random/rule-based generators can be stubbed here if convenient)

---

## 2. PHASE 2 — Defense, Evaluation & Finalization
**Covers Gantt tasks 11–14 (Weeks 3–4)**

### Goal of this phase
Take the adversarial sequences and baseline model from Phase 1 and close the loop: evaluate fragility rigorously, retrain the model adversarially, test robustness against *unseen* attacks, compare strategies, and produce the final report/visualizations.

### Deliverables (build all of these)
1. **Attack evaluation & fragility analysis module** (`evaluation.py`)
   - Prediction drift: `|ŷ_original − ŷ_attack|`
   - Attack success rate: % of attacks exceeding a drift threshold
   - Confidence change (if the model outputs any uncertainty/confidence proxy — otherwise use output variance as a stand-in and state that clearly)
   - Perturbation size vs. drift plot (small-change-big-effect = stronger attack)
   - Scientific plausibility rate of generated attacks
2. **Adversarial training / defense module** (`defense.py`)
   - Combine original + adversarial sequences into an augmented training set.
   - Retrain/fine-tune the Phase 1 baseline model on this augmented set (regularization optional but nice to include, e.g. simple weight decay or an adversarial consistency loss term).
   - Save the "robust model" checkpoint separately from baseline so both can be compared later.
3. **Robustness testing module** (`robustness_test.py`)
   - Generate a **fresh, unseen** batch of adversarial sequences (re-run the MCMC engine on held-out test sequences it hasn't seen before).
   - Evaluate both baseline and robust model on this unseen attack set — this is the key robustness comparison.
   - Optional ablation: does removing the plausibility filter, or using fewer MCMC steps, change robustness gains?
4. **Baseline strategy comparison** (as named in the doc's "Proposed Experimental Comparison")
   - Baseline 1: Random perturbation attack generator (simple, implement quickly)
   - Baseline 2: Rule-based perturbation attack generator (simple fixed substitution rules)
   - Proposed: the Phase 1 MCMC/probabilistic generator
   - Compare all three on: attack success rate, prediction drift, perturbation size, scientific validity rate, diversity of generated attacks, rough computational cost (time/# samples).
5. **Final report / visualization notebook**
   - Summary table of all metrics across baseline vs. robust model, across all three attack strategies.
   - Plots: drift distributions before/after adversarial training; robustness score comparison bar chart; a couple of example adversarial sequences with annotated changes.
   - A short written conclusion: did adversarial training improve robustness, by how much, and what's the main limitation/future extension (e.g. swapping in real cheminformatics validity checks, HMMs/Markov Random Fields per the syllabus mapping in the source doc).

### Explicitly out of scope for Phase 2
- Anything already fully delivered in Phase 1 (don't rebuild the tokenizer/baseline model from scratch — reuse Phase 1 artifacts/checkpoints)

---

## 3. Technical Defaults (use these unless told otherwise when re-uploading)

| Choice | Default |
|---|---|
| Language / framework | Python, PyTorch |
| Environment | Single Colab-style notebook per phase, or a small modular script set — ask which format is preferred only if not specified when re-uploading |
| Dataset | Real dataset if one is provided at upload time; otherwise clearly-labeled synthetic SMILES/formula + regression-target dataset generated in code |
| Model size | Small/fast — this is a research-demo pipeline, not a production model |
| Cheminformatics deps | Avoid hard dependency on RDKit; self-contained token/grammar-based validity checks, with RDKit as an optional enhancement only |
| Output format | Runnable notebook cells + printed/plotted results at each stage, so each phase is independently demonstrable |

## 4. Re-upload Instructions (for future you)
When you come back to build this:
1. Upload this spec file alone (no need to re-attach the original proposal doc or diagram — this file supersedes them).
2. State clearly: **"Build Phase 1"** or **"Build Phase 2"** (Phase 2 assumes Phase 1 artifacts exist — attach the Phase 1 notebook/checkpoint if you have it, otherwise say "assume Phase 1 outputs" and Claude will stub reasonable placeholders).
3. Mention if you now have a real dataset, and attach it — otherwise synthetic data will be used automatically per the defaults above.
4. Everything else in this spec should be treated as fixed scope — no need to re-explain the project.
