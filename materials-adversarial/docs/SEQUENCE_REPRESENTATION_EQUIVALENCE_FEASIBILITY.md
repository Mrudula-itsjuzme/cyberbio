# Frozen-sequence representation-equivalence feasibility

## Model, checkpoint and reproduction

The target is `TransformerRegressorModel`, not TransPolymer weights and not
GraphMPNN.  It predicts polyVERSE polymer bandgap in eV.  The checkpoint,
scaler, tokenizer, vocabulary and split hashes are recorded in
`results/representation_equivalence_sequence/model_manifest.json`.  All 631
stored historical validation predictions reproduced within `1e-6 eV` absolute
tolerance (maximum difference `3.1885e-7 eV`).

## Equivalence semantics and cohort

`src/materials_adv/robustness/equivalence.py` requires sanitized RDKit parsing,
canonical isomeric graph identity, and unchanged dummy-atom attachment-point
signatures.  Failed and duplicate candidates are excluded and logged.  The
cohort is the first 100 eligible sources in the sealed 632-row test split;
there is no train sampling.  It contains 2,907 unique verified variants (median
30 per source, minimum 12).

The families are start-point shift, randomized atom ordering, branch-order
permutation and orientation reversal.  Start-point shifts are tagged
`MIPS_REPEAT_SHIFT_COMPATIBLE`; the other graph-equivalent serializations are
tagged `BROADER_EQUIVALENCE`.  This is a compatibility label, not a novelty
claim.  No repeat multiplication was admitted because changing repeat count is
not canonical graph identity under this validator.

## Drift and failure prevalence

Across variants, mean absolute drift was `0.600256 eV`, median `0.440123 eV`,
P90 `1.336331 eV`, P95 `1.707565 eV`, and maximum `3.252631 eV`.  Source-level
worst-case drift had mean `1.665649 eV`, median `1.534304 eV`, P90 `2.477288 eV`,
P95 `2.895291 eV`, and range `0.677677–3.252631 eV`.

No arbitrary 0.1-eV cutoff is used.  The only interpretable pre-existing scale
is the frozen model's historical test MAE, `0.461916 eV`; all 100 evaluated
sources had worst-case drift above that reference scale.  This prevalence is a
model-error-scale comparison, not a chemical measurement tolerance.

## Repeat/shift and broader equivalence

The MIPS-compatible start-point subset contains 1,992 variants over 100 sources:
mean `0.581116 eV`, median `0.429085 eV`, P95 `1.633724 eV`, max `3.007301 eV`.
The broader set contains 915 variants over 82 sources: mean `0.641925 eV`,
median `0.463727 eV`, P95 `1.799255 eV`, max `3.252631 eV`.  Broader variants
have somewhat larger aggregate tails and contain the overall worst case, while
repeat/start-point shifts alone already expose large failures.

## Graph control

The preserved GraphMPNN control has mean drift `4.81e-8 eV`, median zero, P95
`4.77e-7 eV`, and max `9.54e-7 eV` on its historical control bank.  The frozen
sequence model has mean `0.600256 eV` and P95 `1.707565 eV` here.  The cohorts
are not identical, so this establishes an architectural control contrast, not
a clean-accuracy superiority claim.

## Reliability and ordinary error

At source level, Spearman correlation between clean absolute error and original-
anchored worst drift is `rho=0.0928` (95% source-bootstrap CI `[-0.1219, 0.3060]`).
For centered disagreement it is `rho=0.1424` (CI `[-0.0654, 0.3405]`).  Both
intervals include zero.  AUROC was not run because the repository contains no
pre-existing defensible high-error classification threshold.

Reliability classification: `NO_RELIABILITY_SIGNAL` for this cohort.

## Representation shortcuts and length control

Variant-level Spearman associations with drift were small: token-count delta
`rho=-0.1105`, character-length delta `rho=-0.1105`, tokenization fragmentation
`rho=-0.1566`, start-point indicator `rho=-0.0443`, and orientation indicator
`rho=0.0324`.  The recorded OLS is explicitly descriptive and controls for
token and character-length deltas plus family indicators.  Large drift remains
present among zero-length-delta graph-equivalent serializations; the experiment
does not support a causal interpretation of any coefficient.

Internal embeddings were not added: although the existing model exposes
`encode`, this task prioritized the frozen prediction behavior and required no
architecture changes.  Status: `NOT_RUN`.

## Limitations and gates

RDKit graph identity does not prove equality of every solid-state polymer
microstructure convention.  The MIPS label is limited to serialization
start-point compatibility.  Some sources yielded fewer than 20 unique variants.
Graph and sequence control banks differ.  The OLS table does not include
inferential standard errors and is not causal.

Feasibility classification: `STRONG_EQUIVALENCE_FAILURE`.

Reliability classification: `NO_RELIABILITY_SIGNAL`.

Continuation gate: **CONTINUE** because conditions A and B hold: worst-case
equivalent-representation drift is meaningful relative to the frozen model's
historical error scale, and the broader set has a larger P95 and overall maximum
than the repeat/start-point subset.  Condition C is not supported.  No defense
code is implemented.
