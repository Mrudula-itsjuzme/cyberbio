# Phase 6 invariance-defense preregistration

Status: **FROZEN BEFORE DEFENSE TRAINING**. This document is created before any
Phase 6 optimizer is instantiated or defense checkpoint is trained.

## Prerequisites and immutable inputs

- Baseline checkpoint:
  `results/models/transformer_regressor/model.pt`, SHA-256
  `0b1bb3f62ca72a897fd84c8e7f3129f8a7a9d75c1179c10debb63e46113731e4`.
- Dataset split: `data/processed/splits.json`; train/validation/test contain
  2,946/631/632 rows. The test split remains sealed through configuration and
  lambda selection.
- Final robustness cohort: the frozen 100-source test manifest with SHA-256
  `05a3965e78c926ac123dcd4e433c80989992162cfe24d94b9976af49eb2f9a20`.
- Broader threat model: Phase 5B commit
  `6606e28895558cd26cbbc9cc73e89cf44f27c591`.
- Repeat/shift threat model: Phase 5C implementation commit
  `7a30a7938c121da0b624e41b1e30f62682e47a20`; freeze audit commit
  `2a8ec5c1b216f982946eac9e2a1ffcdb0a4b684b`.
- Tokenizer: `src/materials_adv/data/tokenizer.py`; vocabulary:
  `data/processed/vocab.json`; maximum length 256.
- Every defense variant must pass the corresponding frozen equivalence
  validator, normalize attachment spelling to `[*]`, contain zero unknown
  tokens, and contain at most 256 tokens. No test representation may appear in
  a training or validation bank.

## Hypotheses

- **H1 (primary):** explicit equivalence-consistency training reduces the
  source-level mean worst-case drift, `max |f(x')-f(x)|`, relative to the frozen
  Transformer on the sealed test cohort.
- **H2:** broader-equivalence consistency improves robustness beyond
  repeat/shift-only consistency.
- **H3:** a robustness improvement can be obtained without a clean-test MAE
  increase exceeding the preregistered small-cost bound.
- **H4:** ordinary target-labeled augmentation explains only part of any gain.
- **H5:** a defended model has weaker descriptive drift associations with token
  displacement/order artifacts.

## Initialization and training budget

All primary defenses fine-tune independently from the exact frozen baseline
checkpoint. Retraining from random initialization is out of scope. The fixed
target scaler is loaded and never refit. Primary training seeds are
`20260815`, `20260816`, and `20260817`.

Each selected defense receives five epochs, the same 2,946 source examples per
epoch, one deterministic sampled variant per source per epoch, batch size 16,
AdamW learning rate `1e-4`, weight decay `1e-4`, and identical optimizer-step
budget. Checkpoints are selected by validation data only. Wall time, optimizer
steps, source examples, and variant examples are recorded per run.

## Defenses

1. `baseline`: immutable original checkpoint; no training.
2. `ordinary_augmentation`: variant receives the source property target; no
   consistency term. Variant families are sampled from the broader bank.
3. `repeat_shift_average`: genuine repeat/shift variants and average anchored
   consistency.
4. `broader_average`: broader graph-equivalent variants and average anchored
   consistency.
5. `broader_worst`: a small set of four broader variants is sampled per source;
   the largest squared prediction disagreement is penalized.

`broader_average_plus_augmentation` is not included: it would add another
factor beyond the minimal comparison. The ordinary-augmentation arm already
isolates target-labeled augmentation.

The secondary unseen-family run trains `shift + randomized_atom_ordering` and
holds out `branch_order_permutation`. It uses the selected broader lambda and
seed `20260815`; it is not used to select a primary defense.

## Objective

Targets and predictions are in the baseline model's normalized output space.
For source prediction `p`, variant predictions `p_j`, and normalized target
`y`, average-consistency training uses

`L = MSE(p,y) + lambda * mean_j (p_j-p)^2`.

Ordinary augmentation uses `0.5*(MSE(p,y)+MSE(p_j,y))`. Sampled-worst training
replaces the mean consistency term by `max_j (p_j-p)^2`. No variant label is
invented beyond the equivalence assumption already validated by the frozen
protocol. Set-centered loss is omitted to keep the comparison minimal.

## Lambda selection and test-once rule

The grid is `lambda in {0.01, 0.1, 0.5, 1.0}`. This spans two orders of
magnitude around equal weighting because property and consistency losses share
the normalized prediction scale. Lambda screening uses seed `20260815` and the
validation split only.

For each curriculum, candidates with clean validation MAE no more than
`0.02 eV` above the frozen baseline validation MAE (`0.460148 eV`) are eligible.
Among eligible candidates, select the lowest mean validation source worst-case
drift. If none is eligible, select the lowest validation MAE and classify the
arm as failing the clean-cost gate. Ties within `0.005 eV` robustness use lower
clean MAE, then lower lambda. The selected lambda is frozen before the three
training-seed checkpoints are evaluated on test.

## Validation and test banks

Training and validation banks are generated deterministically from their own
split rows with seeds derived from dataset row index. The validation bank is
used for lambda selection. The two committed Phase 5B/5C test banks are reused
unchanged for the one final evaluation. Test targets or predictions never enter
selection.

## Tokenizer intervention

The historical behavior maps unknown tokens, including RDKit bare `*`, to input
ID 0, which is also padding. The corrected pipeline normalizes bare `*` to the
known `[*]` token and rejects any remaining unknown. This is a preprocessing fix,
not a trained defense. Its contribution is reported separately using the frozen
baseline before/after Phase 5B predictions. Clean source inputs already use
`[*]`, so no clean-model retraining is attributed to this fix.

## Positional mechanism ablation

A single secondary ablation disables addition of learned absolute positional
embeddings while retaining the same Transformer layers and dimensions. It is
fine-tuned with the selected broader-average protocol at seed `20260815` and is
reported as a mechanism test, not a new architecture family or primary defense.

## Metrics and statistics

Clean sealed-test metrics are MAE, RMSE, and R2 on all 632 test rows. Robustness
is evaluated on the fixed 100-source broader cohort and the 99 applicable
repeat/shift sources: source worst-case mean, median, P90, P95, and maximum,
plus per-family results. Primary paired inference uses source-level baseline
minus defense worst-case drift, a 5,000-replicate source bootstrap CI, paired
Wilcoxon test, and paired standardized mean change. Training-seed variation is
reported separately.

Shortcut analyses are descriptive Spearman associations with position
displacement, token-count change, and family/order indicators. Representation
disagreement is measured but is not interpreted as a reliability signal.

## Success classifications

A robustness improvement requires a positive paired mean reduction whose 95%
bootstrap CI excludes zero on the broader threat model. Clean-cost classes use
the three-seed mean test MAE change, while the thresholds were fixed here from
the validation/reference scale:

- `ROBUSTNESS_IMPROVED_NO_CLEAN_COST`: improvement and MAE increase <=0.005 eV.
- `ROBUSTNESS_IMPROVED_SMALL_CLEAN_COST`: improvement and increase >0.005 but
  <=0.020 eV.
- `ROBUSTNESS_IMPROVED_LARGE_CLEAN_COST`: improvement and increase >0.020 eV.
- `REPEAT_SHIFT_ONLY_ROBUSTNESS`: supported repeat/shift reduction without
  supported broader reduction.
- `BROADER_INVARIANCE_GENERALIZES`: broader reduction plus supported reduction
  on the preregistered held-out branch-order family.
- `NO_MEANINGFUL_DEFENSE_BENEFIT`: the primary improvement criterion fails.
- `UNSTABLE`: seed outcomes disagree in sign or show unacceptable dispersion.

The paper claim is allowed only if broader-equivalence reduction is supported,
all three selected-seed runs improve mean source worst-case drift, and the clean
cost is at most `0.020 eV`. No attack or defense specification may be changed
after final test evaluation begins.
