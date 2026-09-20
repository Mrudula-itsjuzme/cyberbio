# Genuine repeat/shift invariance results

This report is generated for Phase 5C from the frozen Phase 5B cohort and model.
It does not modify the validated Phase 5B bank, retrain a model, optimize an
attack, or implement a defense.

## Representation semantics and protocol

The input is a finite, two-`[*]` PSMILES repeat cell. `SHIFT` moves the periodic
cell boundary along the endpoint-to-endpoint backbone. `REPEAT` tiles the cell
head-to-tail twice. `SEGMENTATION` translates the doubled cell. Validation
requires endpoint semantics, equal endpoint bond type, formal-charge scaling,
isomeric periodic-graph identity or exact tiled-graph identity, tokenizer
round-trip, zero unknown tokens, and length at most 256. See
`REPEAT_SHIFT_REPRESENTATION_FORENSICS.md` and
`MIPS_PROTOCOL_INTERPRETATION.md`.

## Implementation and cohort

`src/materials_adv/robustness/repeat_shift.py` contains transformation and
validation code. `scripts/run_repeat_shift_invariance.py` reuses the 100 Phase 5B
source IDs and the frozen Transformer checkpoint. Non-applicable sources remain
in `cohort_applicability.csv`; no replacement source is selected.

Ninety-nine of the unchanged 100 sources are applicable. The protocol produced
1,181 strict variants; all 56 exclusions and the one non-applicable source are
retained rather than replaced. `variants.csv` contains only primary variants.

## Transformer results

Across variants, mean/median/P90/P95/max drift is
`0.528685/0.408635/1.191840/1.450202/2.859155 eV`. Source worst-case
mean/median/P90/P95/max is
`1.117856/0.977281/1.853790/2.120893/2.859155 eV`. Source-bootstrap 95% CIs are
`[1.001543, 1.231960]` for the mean, `[0.812231, 1.179380]` for the median, and
`[1.671316, 2.119493]` for P90. Bootstrap samples polymers, not variants.

By family: `SHIFT` has 77 sources/654 variants, median `0.341575`, P95
`1.166140`, max `2.634227 eV`; `REPEAT` has 92/92, median `0.433026`, P95
`1.518306`, max `1.927115 eV`; `SEGMENTATION` has 91/435, median `0.558659`,
P95 `1.749886`, max `2.859155 eV`. No separate orientation family was justified.

## Graph control

The matched GraphMPNN is evaluated on the same strings in `graph_control.csv`.
Unlike the Phase 5B serialization control, true boundary translations change
the finite dummy-atom graph presented to this non-periodic GraphMPNN. Nonzero
drift therefore does not refute periodic equivalence and the model is not
described as a periodic-invariance control. Its matched mean/median/P95/max
drift is `0.181440/0.159080/0.444026/0.679655 eV`.

These results preserve two distinct claims: (A) Phase 5B GraphMPNN predictions
are essentially invariant to ordinary graph-identical serialization
permutations; (B) Phase 5C repeat/periodic transformations are not a directly
comparable invariance control because this GraphMPNN does not encode periodic
closure.

## Repeat versus shift and position analysis

Families are never pooled without separate reporting. For `SHIFT`, token
displacement has descriptive Spearman rho `0.155337` with Transformer drift;
this is not a causal estimate.

## Broader equivalence, matched k, and set inclusion

`repeat_shift_vs_broader.csv` compares source worst-case and median drift with
the unchanged Phase 5B strict bank on the same applicable sources. It records
overlap, unique broader variants, and one of `REPEAT_SHIFT_SUFFICIENT`,
`BROADER_EQUIVALENCE_NEEDED`, `TIE`, or `NOT_COMPARABLE`.
The original `matched_k_comparison.csv` is retained as a secondary,
changing-cohort diagnostic: eligible source counts fall from 99 at k=1 to 24 at
k=20. Its k=10-to-k=20 decline therefore compared different populations and is
not a valid monotonic search-size curve.

The frozen primary comparison is `matched_k_fixed_cohort.csv`. It holds the 24
sources with at least 20 variants in both banks fixed at every k and uses 5,000
deterministic nested subsamples without replacement. Mean expected worst drift
for repeat-shift versus broader is: k=1 `0.419580/0.468425`, k=2
`0.579210/0.678291`, k=5 `0.784897/0.999580`, k=10
`0.921969/1.273515`, and k=20 `1.042272/1.561256 eV`. Both series are monotone.
The broader-minus-repeat difference increases from `0.048845` to `0.518984 eV`.

The broader bank has a higher observed source worst case for 79/99 (`79.80%`)
of applicable sources; repeat/shift is higher for 20/99. Eleven exact strings
overlap, but no genuine repeat/shift variant is the exact Phase 5B worst variant.
Thus the inclusion classification is 79 `BROADER_EQUIVALENCE_NEEDED` and 20
`REPEAT_SHIFT_SUFFICIENT`.

## Failure casebook

Eight auditable examples, including five high- and three low-drift pairs, are in
`REPEAT_SHIFT_SANITY_CASEBOOK.md`. Strings, token sequences, predictions,
equivalence rationale, and matched graph predictions are shown.

## Limitations

The paper protocol is underspecified at code level; the dataset cannot establish
3D periodic-cell equivalence; shortest-path backbone selection can exclude
alternative paths in cyclic cells; and the legacy GraphMPNN lacks periodic
closure. Results apply to this frozen checkpoint and sealed cohort.

## Gates

Replication relationship: `PARTIAL_REIMPLEMENTATION`. The representation
transform is a faithful reimplementation, but this is not the MIPS training or
loss-maximizing evaluation protocol.

Novelty gate: `B. BROADER_EQUIVALENCE_ADDS_SUBSTANTIAL_FAILURES`. Genuine
repeat/shift failure is real, while the broader bank is worse for 79.80% of
matched sources and remains worse at each matched k on average.

Defense gate: `BROADER_EQUIVALENCE_DEFENSE_REQUIRED`. This is only the justified
next question; no defense is implemented in Phase 5C.
