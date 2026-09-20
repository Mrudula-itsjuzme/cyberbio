# Representation threat-model freeze

No defense is implemented by this freeze. It locks two distinct evaluation
protocols so later work cannot silently mix their semantics, cohorts, or search
spaces.

## Shared frozen state

- Model: frozen two-layer `TransformerRegressorModel`.
- Checkpoint: `results/models/transformer_regressor/model.pt`.
- Cohort: the same 100 held-out sources in
  `results/representation_equivalence_sequence/cohort_manifest.json`.
- Cohort manifest SHA-256:
  `05a3965e78c926ac123dcd4e433c80989992162cfe24d94b9976af49eb2f9a20`.
- No training, attack optimization, cohort substitution, or defense is allowed
  inside either frozen evaluation.

## Threat model A: faithful repeat/shift protocol

- Defining commit: `7a30a7938c121da0b624e41b1e30f62682e47a20`.
- Protocol status: `FAITHFUL_REIMPLEMENTATION`; relationship to the full MIPS
  experiment: `PARTIAL_REIMPLEMENTATION`.
- Cohort: all 100 shared sources are assessed for applicability; 99 are
  applicable and `poly_2923` remains explicitly `NOT_APPLICABLE`.
- Families: `SHIFT`, `REPEAT`, and `SEGMENTATION` (repeat then translation).
  No orientation family is included.
- Counts: 654 SHIFT, 92 REPEAT, 435 SEGMENTATION; 1,181 total strict variants.
- Variant artifact SHA-256:
  `27a2ace941503b1d64481272c447848d4c56a6b69c6676ab55f7989d40308145`.
- Strict eligibility: verified two-endpoint attachment semantics; matching
  endpoint bond types; periodic-graph or exact tiled-graph validation; formal
  charge and stereochemistry preservation; exact tokenizer round-trip; zero
  unknown tokens; at most 256 tokens; no truncation.
- Variant drift mean/median/P90/P95/max:
  `0.528685/0.408635/1.191840/1.450202/2.859155 eV`.
- Source worst-case mean/median/P90/P95/max:
  `1.117856/0.977281/1.853790/2.120893/2.859155 eV`.
- Limitations: only 1D periodic topology is validated; the accessible MIPS
  description does not fix every sampling choice; the 20-variant cap and
  shortest backbone path are repository choices; the legacy GraphMPNN does not
  encode periodic closure.

## Threat model B: broader graph-equivalent serialization

- Defining commit: `6606e28895558cd26cbbc9cc73e89cf44f27c591`.
- Cohort: all 100 shared held-out sources; none substituted.
- Families: start-point serialization, randomized atom ordering, branch-order
  permutation, and orientation reversal.
- Counts: 1,992 start-point, 421 randomized-order, 419 branch-order, and 75
  orientation variants; 2,907 total strict variants.
- Strict-pair artifact SHA-256:
  `236d300496752a9cc146e964e604fb883629be3f65246c974cf4e7987c9197df`.
- Strict eligibility: verified finite graph, formula, charge, connected
  components, attachment semantics, and stereochemistry; exact normalized
  tokenizer input; zero unknown tokens; at most 256 tokens; deterministic and
  batch-invariant inference.
- Variant drift mean/median/P90/P95/max:
  `0.579362/0.409394/1.314319/1.708383/3.815713 eV`.
- Source worst-case mean/median/P90/P95/max:
  `1.684843/1.544650/2.402767/3.087643/3.815713 eV`.
- Limitations: these are finite graph-equivalent serializations, not genuine
  MIPS repeat/shift transformations; family bank sizes differ; worst-case
  comparisons require matched-k control.

## Frozen matched-k comparison

The primary comparison uses the 24 sources having at least 20 variants in both
threat models. For every source and each of 5,000 deterministic repetitions,
variants are permuted once without replacement and k uses the first k entries.
This nesting guarantees that each source/repetition maximum cannot decrease.

| k | Sources | Repeat/shift expected max | MC std | Broader expected max | MC std | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0.419580 | 0.062122 | 0.468425 | 0.086428 | 0.048845 |
| 2 | 24 | 0.579210 | 0.059674 | 0.678291 | 0.091256 | 0.099082 |
| 5 | 24 | 0.784897 | 0.048128 | 0.999580 | 0.093603 | 0.214683 |
| 10 | 24 | 0.921969 | 0.036130 | 1.273515 | 0.088724 | 0.351547 |
| 20 | 24 | 1.042272 | 0.001719 | 1.561256 | 0.057176 | 0.518984 |

The earlier changing-cohort table remains a secondary diagnostic. Its source
counts are 99, 97, 76, 60, and 24, causing the apparent k=20 decline.

## GraphMPNN wording freeze

1. Phase 5B establishes that GraphMPNN is essentially invariant to ordinary
   graph-identical serialization permutations (maximum numerical drift about
   `4.77e-7 eV`).
2. Phase 5C GraphMPNN scores on repeat/periodic strings are not a periodic
   invariance control. Its finite dummy-atom representation does not encode the
   periodic closure used to prove polymer equivalence. Nonzero scores must not
   be presented as evidence against graph permutation invariance or as a
   model-superiority comparison.

## Frozen conclusion

The 79/99 broader-worse comparison is unchanged. The fixed-cohort matched-k
result is monotone and supports the same novelty gate:
`B. BROADER_EQUIVALENCE_ADDS_SUBSTANTIAL_FAILURES`. The future defense question
remains `BROADER_EQUIVALENCE_DEFENSE_REQUIRED`; it is outside this freeze audit.
