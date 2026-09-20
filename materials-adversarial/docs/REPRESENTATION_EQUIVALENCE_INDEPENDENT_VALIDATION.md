# Independent validation of sequence representation failures

## Original result and adversarial objective

Phase 5 reported strong representation drift. Phase 5B attempted to invalidate it through stricter chemistry, tokenizer, truncation, determinism, batching, masking and matched-graph controls without retraining, defense work, cohort expansion or new search.

## Chemical and input audit

All 2,907 pairs preserve graph, formula, charge, components, stereochemistry and attachment semantics. The raw RDKit variants nevertheless contain two unknown bare-star tokens each because the vocabulary expects `[*]`. This invalidates the original numerical estimates as clean tokenizer-controlled estimates. After attachment spelling normalization, all 2,907 same-bank variants have zero unknowns and remain chemically identical.

No sequence exceeds 256 tokens. Inference is deterministic, and maximum batch-control variation is `7.15e-7 eV`. Matched GraphMPNN maximum drift is `4.77e-7 eV`.

## Strict subset

The strict subset contains 100 sources and 2,907 variants. Variant drift: mean `0.579362`, median `0.409394`, P90 `1.314319`, P95 `1.708384`, maximum `3.815713 eV`. Source worst-case: mean `1.684843`, median `1.544650`, P90 `2.402767`, P95 `3.087643`, maximum `3.815713 eV`.

Family P95/max values are: start-point `1.658034/3.815713`, randomized order `1.923355/3.632825`, branch order `1.877607/3.198632`, orientation `2.153961/3.551546 eV`. All families survive independently.

The MIPS-compatible subset is empty: the implemented start-root operation is serialization shift, not a faithful repeat-unit-shift protocol. The broader verified-equivalence result remains strong.

## Search-size and uncertainty

Expected source worst-case drift rises with k: `0.5867` (1), `0.8059` (2), `1.1031` (5), `1.3423` (10), `1.5725` (20), `1.6848 eV` (30). Source-bootstrap 95% CIs are mean `[1.5643,1.8175]`, median `[1.4226,1.6822]`, and P90 `[2.3179,3.0835]`.

All sources exceed the historical aggregate MAE (`0.4619 eV`) and 99% exceed aggregate RMSE (`0.6235 eV`); these are descriptive scale comparisons, not per-source failure thresholds.

## Headline cases and root cause

The corrected strict failures survive: `poly_29` `3.815713 eV`, `poly_443` `3.551546 eV`, and `poly_3050` `3.165809 eV`; each has zero unknowns, no truncation, deterministic/batch-invariant Transformer predictions and zero matched-graph drift at displayed precision.

Root-cause classification: `MULTIPLE_FACTORS`, specifically a confirmed unknown-token artifact in the original raw bank plus surviving `SEQUENCE_ORDER_SHORTCUT`/`POSITIONAL_ENCODING_SENSITIVITY` in the corrected strict bank. No padding-mask bug is supported.

Revised feasibility classification: `STRONG_EQUIVALENCE_FAILURE_CONFIRMED`. The original exact metrics are superseded by strict metrics, but the conclusion survives.

Next gate: independently implement a true repeat-unit/MIPS protocol and preregister a defense comparison. No defense is implemented here.
