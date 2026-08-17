# ARCHITECTURE FORENSIC

Adversarial audit of the Phase 1–2C results. The goal was to **break** the
existing conclusions, not defend them.

Run `.venv/bin/python scripts/forensic_diagnostics.py` to reproduce every number
here. Output is persisted to `results/forensic_diagnostics.json`.

Date: 2026-08-15. No models were retrained. No previous conclusion was deleted;
superseded claims are marked in place.

---

## 0. Headline

**The Phase 1F conclusion — "length-changing perturbations cause the ~30 K drift"
— does not survive.** A length-*preserving* control that changes no atoms, no
bonds and no molecular formula produces drift of the **same magnitude**
(36–39 K vs ~30 K), in **all five seeds**.

The model is not specifically length-sensitive. It is **globally unstable to any
token-level perturbation**, and the Phase 1E/1F comparison mistook an
*edit-severity* confound for a *length* mechanism.

Separately: a **length-only linear regression beats the Transformer on
validation** (76.77 K vs 79.04 K). The 30 K drift is drift of a model that has
not demonstrably learned chemistry.

---

## 1. Exact architecture as implemented

Traced from [transformer.py](../src/materials_adv/models/transformer.py),
[regression.py](../src/materials_adv/models/regression.py),
[train.py](../src/materials_adv/training/train.py).

```
PSMILES string
  -> tokenize()            regex, first-match; 45-token vocab
  -> char2idx[t] = i+1     id 0 reserved for PAD; OOV also -> 0  (COLLISION, §4.1)
  -> nn.Embedding(46, 64, padding_idx=0)
   +  nn.Embedding(256, 64)          LEARNED absolute positional encoding
  -> nn.TransformerEncoder(2 x TransformerEncoderLayer)
       d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
       activation=ReLU (torch default), norm_first=False (POST-LN, torch default)
  -> masked mean pooling over non-pad positions
  -> nn.Linear(64, 1) -> squeeze
  -> scaler.inverse_transform()  ->  Tg in Kelvin
```

| Question (Part 2) | Answer as implemented |
|---|---|
| 1. Embedding | `nn.Embedding(vocab+1=46, 64)`, `padding_idx=0` |
| 2/3. Positional encoding | **Learned** absolute, `nn.Embedding(256, 64)`. Not sinusoidal |
| 4. Sequence length changes | Positions are re-derived as `arange(seq_len)`; an insertion at index *j* **re-indexes every downstream token** |
| 5. Pooling | Masked mean over non-pad positions |
| 6. `[CLS]` | Supported in code, **not used**. `pooling: mean` in config |
| 7. Mean pooling | Yes — this is the active path |
| 8. Padding in pooling | **No.** Correctly excluded — verified (§3, D1) |
| 9/10. Masks | **Correct.** Verified numerically exact to 3e-05 |
| 11. Dimensionality | `[B,L] -> [B,L,64] -> [B,L,64] -> [B,64] -> [B]` |
| 12. Layers/heads/params | 2 layers, 4 heads, **86,337 params**, trained on **204 samples** |
| 13. Activation | ReLU (torch default, never overridden) |
| 14. Dropout | 0.1 inside encoder layers only. **None in the regression head** |
| 15. Normalization | LayerNorm, **post-norm** (torch default) |
| 16. Regression head sees | A single 64-d masked-mean vector — no length feature, no pooled statistics |

### Mechanisms that could cause insertion/deletion sensitivity

1. **Learned absolute positional encoding + re-indexing.** An insertion at
   position *j* shifts every subsequent token onto a different learned
   positional vector. With 204 training samples, positional embeddings beyond
   the common length range are barely trained.
2. **Mean pooling is a 1/L average.** A single edited token contributes 1/L of
   the pooled vector, so short sequences are structurally more perturbable. This
   predicts drift ∝ 1/L — **and that prediction is confirmed** (§3, D5).
3. **Post-LN with 2 layers and no warmup** is a known instability, though not
   isolated here.

---

## 2. Data and attack pipeline

**Data.** `final_polymer_properties_fromliterature.csv` (741 rows) → 443 non-null
Tg → RDKit canonicalization → `drop_all` on 31 conflicting duplicates → **247
samples**. Scaffold split → **train 204 / val 37 / test 6**. `TargetScaler`
(mean 330.713, std 88.046) fitted on training targets only.

**Attacks.** Token-level edits with protected positions (attachments, ring
closures, branches) → RDKit representation validity → plausibility → predict →
signed drift. 5 candidates/sample/family; invalid candidates excluded from drift
statistics.

---

## 3. Diagnostics run, and what they showed

### D1 — Padding/masking is CORRECT (hypothesis D refuted)

Appending PAD-masked positions is a true no-op:

| Extra masked pads | +0 | +5 | +20 | +100 |
|---|---|---|---|---|
| max abs drift (K) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Predictions are also **batch-invariant** and **padding-width-invariant** (max
deviation 3.05e-05, float32 noise). **Hypotheses D (masking bug), C (pooling
artifact) and Q (evaluation bug) are refuted for the padding path.**

### D2 — Position shift alone is NOT sufficient

Unmasked PAD tokens change length and re-index positions while adding no
chemistry:

| Unmasked pads appended | +1 | +3 | +10 |
|---|---|---|---|
| mean abs drift (K) | 0.35 | 4.99 | 5.59 |

A pure positional/length shift buys **~5 K**, not 30 K. **Hypothesis E
(positional-encoding artifact) is real but small — it explains at most ~1/6 of
the effect.**

### D3 — Control baselines: the model is not clearly learning chemistry

MAE in Kelvin. Test n=6 — treat test numbers as anecdote.

| Model | Val MAE | Test MAE |
|---|---|---|
| Mean predictor | 88.48 | 132.54 |
| Median predictor | 87.53 | 130.67 |
| **Length-only linear regression** | **76.77** | **76.39** |
| Bag-of-tokens ridge (α=1) | 79.59 | **43.24** |
| **Transformer (Phase 1)** | **79.04** | 52.02 |

**A one-parameter model on sequence length alone beats the Transformer on
validation.** Bag-of-tokens — no order, no positions — also matches it, and beats
it on test. **Hypothesis T (hidden length/Tg correlation) is CONFIRMED**, and
this reframes every drift number in the project.

The split is badly confounded:

| Split | n | mean Tg (K) | mean token length |
|---|---|---|---|
| train | 204 | 330.7 | 22.6 |
| val | 37 | 384.5 | 43.5 |
| test | 6 | 463.2 | 56.7 |

corr(length, Tg) = **0.372** (Spearman 0.378, p=8e-10) across all 247 samples.
Scaffold splitting sorted longer, higher-Tg polymers into val/test. Test Tg
values (418.7–496.0 K) lie **entirely above** the training mean — the test set is
not representative, and negative R² everywhere follows directly. **Hypothesis K
(scaffold-split instability) is CONFIRMED.**

Note the Transformer's predictions correlate with length at only 0.006 on
validation while truth correlates at 0.209 — it is not even exploiting the one
signal that works.

### D4/D7 — THE DECISIVE RESULT: length is not the mechanism

Length-preserving controls vs a length-changing control, mean abs drift (K),
across all 5 Phase 2C baseline seeds:

| Seed | shuffle (len same, composition same) | reverse (len same, composition same) | duplicate +1 token (len +1) |
|---|---|---|---|
| 20260815 | 36.03 | 38.58 | 38.66 |
| 20260816 | 40.47 | 37.13 | 24.92 |
| 20260817 | 20.07 | 17.61 | 18.80 |
| 20260818 | 24.60 | 23.35 | 18.68 |
| 20260819 | 39.07 | 35.51 | 23.72 |
| **mean** | **32.05** | **30.44** | **24.96** |

Shuffle and reverse **preserve length exactly, preserve the token multiset
exactly, and change no atom, bond or formula.** They produce drift *at least as
large* as length-changing edits, in every seed.

**This falsifies the Phase 1F causal claim.** Phase 1E's "length-preserving"
families were mis-specified: `rearrangement` permutes only a 3-token window
(often a near-no-op), and `substitution` was constrained to swap within tight
chemical roles. Neither is a strong order perturbation. Once a genuine
length-preserving perturbation is applied, drift is as large as insertion's.

### D5 — Mean-pool dilution CONFIRMED (hypothesis C, as a real property)

Drift from a single-token edit vs sequence length:

| Token length | n | insertion drift | deletion drift |
|---|---|---|---|
| [0,15) | 3 | **75.34** | 46.12 |
| [15,30) | 8 | 54.72 | 40.11 |
| [30,60) | 18 | 26.76 | 17.41 |
| [60,999) | 8 | 19.25 | 22.01 |

corr(length, insertion drift) = **−0.502**; corr(1/length, insertion drift) =
**+0.575**. Drift scales as **1/L**, exactly as masked mean pooling predicts.
Per-sample drift is therefore **not comparable across sequence lengths**, and
because val/test sequences are systematically longer than train, drift
statistics are confounded with the split.

### D6 — Attack families are NOT matched in severity (hypothesis G CONFIRMED)

RDKit audit of the 330 valid Phase 2B baseline candidates:

| Family | n valid | validity rate | mean abs ΔMW | formula preserved | identical molecule | mean abs drift |
|---|---|---|---|---|---|---|
| substitution | 88 | 48.6% | 10.79 | 0% | 0% | 12.97 |
| insertion | 85 | 45.9% | 15.75 | 12.9% | **12.9%** | 30.67 |
| deletion | 94 | 50.8% | 12.28 | 2.1% | 1.1% | 30.73 |
| **rearrangement** | 63 | 63.0% | **0.00** | **100%** | 1.6% | 7.99 |

**Rearrangement preserves the molecular formula in 100% of cases and changes
molecular weight by exactly zero** — it permutes atoms within a window, an
isomerization. Insertion/deletion add or remove heavy atoms. These are not the
same experimental treatment, so the Phase 1E "attack mechanism" comparison
**confounds mechanism with chemical severity**. The correct reading of Phase 1E
is "bigger chemical changes produce bigger drift", which is far less interesting.

Also: **12.9% of "insertion" attacks produce a molecule identical to the
original** after canonicalization — those are no-ops scored as attacks.

---

## 4. Competing explanations — verdicts

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| A | True model vulnerability | **PARTLY** | Model is genuinely unstable, but to *all* perturbation, not to length |
| B | Position-shift artifact | **SMALL** | D2: ~5 K of ~30 K |
| C | Mean-pooling / length artifact | **CONFIRMED** | D5: drift ∝ 1/L, r=+0.575 |
| D | Padding/masking bug | **REFUTED** | D1: exact no-op |
| E | Positional-encoding artifact | **SMALL** | D2 |
| F | Tokenization artifact | **REFUTED** | 45-token vocab, round-trip tested, no OOV in data |
| G | Attacks chemically drastic | **CONFIRMED** | D6: rearrangement ΔMW=0 vs insertion 15.75 |
| H | RDKit-valid but nonsensical | **CONFIRMED** | Valid ≠ plausible; already acknowledged |
| I | One-token distribution shift | **LIKELY** | 204 training samples |
| J | Dataset sparsity | **CONFIRMED** | 247 samples, 86k params |
| K | Scaffold-split instability | **CONFIRMED** | D3: train/val/test Tg 330/384/463 K |
| L | Small-data overfitting | **CONFIRMED** | Negative R² everywhere |
| M | Model undercapacity | **REFUTED** | 86k params on 204 samples is over-, not under-capacity |
| N | Regression-head instability | **NOT ISOLATED** | Single Linear, no dropout |
| O | Target-normalization interaction | **RESOLVED IN 2B** | Already controlled |
| P | Random-search selection bias | **PRESENT** | §5 |
| Q | Evaluation bug | **REFUTED** | D1/D2 batch- and pad-invariant |
| R | Drift inflated by poor calibration | **CONFIRMED** | Baseline predictions are worse than a length-only model |
| S | Length AND composition both change | **CONFIRMED** | D6 |
| T | Hidden length/Tg correlation | **CONFIRMED** | r=0.372, and length-only beats the Transformer |

---

## 5. Statistical weaknesses (Part 9)

- **Budget is not matched.** Rearrangement yielded 100 raw candidates vs 185 for
  insertion/deletion, because many sequences lack an eligible 3-token window.
  Comparing families at unequal budgets biases max-drift statistics.
- **Max drift is unstable** and scales with candidate count — it is the worst of
  *n* draws. The one Phase 2C effect that "survived" (insertion max drift) is
  precisely the statistic most sensitive to budget, and to the 1/L dilution
  effect.
- **Validity-conditioned drift is a survivorship filter.** ~50% of candidates are
  discarded, and the discard rate differs by family (45.9%–63.0%), so the
  surviving populations are not comparable.
- **n=6 test set.** Every clean-MAE comparison is anecdote.
- **Candidates are not independent** — already correctly noted in Phase 2C.

---

## 6. Target-preservation — revised taxonomy (Part 10)

The current assumption (adversarial PSMILES keeps the original Tg) is defensible
for only one of the four families.

| Class | Families | Justification |
|---|---|---|
| **A. Representation-level** | *none currently* | No family is a pure re-encoding of the same molecule |
| **B. Chemistry-changing** | substitution, insertion, deletion | Change formula and heavy-atom count — true Tg almost certainly differs |
| **B′. Isomerization** | rearrangement | Formula preserved (100%), but connectivity changes — a different polymer with different Tg |
| **C. Invalid** | ~50% of all candidates | Filtered |

**No current attack family is a legitimate representation-level perturbation.**
The label-preserving assumption underpinning adversarial training in 2A/2B/2C is
therefore chemically unjustified for *every* family, not just the length-changing
ones.

A genuine representation-level attack exists and is not implemented: **SMILES
randomization** — different valid SMILES writings of the *identical* molecule
(RDKit `doRandom=True`). Tg is provably unchanged, so any drift is pure model
error. This is the correct control, and the correct augmentation.

---

## 7. Revised claims

### Justified

1. Padding, masking, batching and the scaler are correctly implemented.
2. The model is unstable under token-level perturbation: ~20–40 K drift.
3. Sensitivity is **not specific to length-changing edits** — length-preserving
   shuffles produce equal or larger drift in all 5 seeds.
4. Per-edit drift scales ~1/L, a direct consequence of mean pooling.
5. Attack families differ in chemical severity; Phase 1E does not isolate
   mechanism.
6. Phase 2C stands: the 2B robustness effects do not replicate.

### NOT justified

1. ❌ "Length-changing perturbations cause the vulnerability" — **refuted (D4)**.
2. ❌ "Positional encoding is the mechanism" — at most ~5 K of ~30 K (D2).
3. ❌ "The Transformer learned polymer chemistry" — a length-only linear model
   beats it on validation.
4. ❌ Any clean-MAE claim from n=6.
5. ❌ "Adversarial training improves chemical invariance" — no family is
   chemistry-preserving.
6. ❌ Cross-family drift comparisons at unequal budgets.

---

## 8. Open scientific questions

1. **Is the vulnerability positional or representational?** D2 says positional
   encoding gives ~5 K, shuffling ~32 K. So the model is sensitive to *token
   context*, not position per se. *Experiment:* retrain with positional encoding
   removed; if drift persists, position is exonerated entirely.
2. **Does the model use chemistry at all beyond length and token counts?**
   *Experiment:* bag-of-tokens vs Transformer under matched CV. Already
   suggestive: they tie.
3. **Is drift just 1/L?** *Experiment:* regress drift on 1/L and report residual
   variance. If R² is high, "drift" is largely a pooling artifact.
4. **Does SMILES randomization — a provably Tg-preserving perturbation — produce
   drift?** This is the cleanest possible robustness test and is currently
   missing. Non-zero drift is unambiguous model error.
5. **How much of every result is the split?** *Experiment:* k-fold scaffold CV.
   Split variance is entirely unmeasured and likely exceeds seed variance.
6. **Is max-drift reduction real or a budget artifact?** The one surviving 2C
   effect is the statistic most sensitive to candidate count.
7. **Are high-drift examples concentrated in short sequences?** D5 says yes —
   is that fully explanatory?
8. **Does mean pooling itself cause the instability?** *Experiment:* compare
   mean vs CLS vs max pooling for drift under fixed data.
9. **Is a Transformer justified at n=204?** XGBoost on descriptors is the
   honest control, and OpenPoly's own paper reports it winning.

---

## 9. Prioritized experiments (Part 12)

### TOP 3 — highest information per unit cost

1. **SMILES-randomization invariance test.** The only perturbation with a
   *provably* unchanged target. Cheap (no training), and it converts "drift" from
   an assumption-laden quantity into unambiguous model error. Resolves Q4 and
   repairs the target-preservation hole.
2. **Positional-encoding ablation** (retrain 5 seeds, no positional embedding).
   Directly tests the mechanism Phase 1F speculated about. ~10 min on CPU.
3. **k-fold scaffold cross-validation** for the length-only / bag-of-tokens /
   Transformer trio. Replaces the n=6 test set and answers "is the Transformer
   learning anything?" with an actual error bar.

### TOP 5 — add

4. **Drift ~ 1/L regression** on existing records. Pure analysis, minutes.
5. **Budget-matched attack comparison** — equal candidate counts per family,
   reporting median and validity rate, not max.

### Do NOT waste time on

- More attack families (Phase 3) — the taxonomy is unsound until §6 is fixed.
- Retraining the defended model — Phase 2C already showed the effect is noise.
- Multi-step / MCMC / probabilistic attacks — compounding an unresolved
  confound.
- Graph models — a large build that does not answer any question above. It is a
  *later* control, not a fix.
- Chasing the >52.02 K threshold — derived from a 6-sample MAE.

---

## 10. Limitations of this forensic audit

1. Shuffle and reverse are chemically absurd strings; they are **mechanistic
   probes of the model's function**, not proposed attacks. They test what the
   model responds to, not what an attacker would do.
2. Diagnostics run on the validation split (n=37). The test set stayed sealed.
3. No retraining, so positional-encoding and pooling claims are *observational*.
   The ablations in §9 are what would make them causal.
4. Bag-of-tokens/length baselines use a single split, and inherit exactly the
   split fragility they are used to criticize.
