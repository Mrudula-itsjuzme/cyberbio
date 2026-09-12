# Phase 5 — Shortcut and Representation Attribution

## Scope and evidence boundary

This phase is diagnostic, not a state-of-the-art benchmark. It uses the active
`polyverse_bandgap` lineage, its existing random split, and existing checkpoints.
No Transformer was retrained. All numbers below are validation-set observations,
not sealed-test estimates and not evidence of causal chemical reasoning.

Evidence labels used here:

- **Observed:** directly recorded in `results/representation_attribution/`.
- **Supported:** an interpretation supported by more than one controlled result,
  but still bounded by this dataset, split, model, and sample.
- **Speculative:** plausible but not established by these experiments.

## Methods

The full validation split contains 631 examples; the training split contains
2,946 examples. Lightweight ridge models use a fixed `alpha=1.0`. The controls
are a training-target mean, token length, token counts, length-normalized token
frequencies, and nine simple RDKit descriptors. Descriptor validity was 2,946 / 2,946
for training and 631 / 631 for validation.

Two existing Transformers are evaluated: `transformer_regressor` and the
historical `transformer_defended`. Their checkpoint SHA-256 hashes and common
target-scaler provenance are in `reproducibility.json`.

For a seeded random sample of 100 molecules, RDKit generated up to 10 randomized
SMILES; the original representation was included. Ninety-nine molecules had 11
total representations and one had 9. All 2,196 model/representation records were
RDKit-valid and canonically equivalent to their source. The same candidate bank
was used for both models.

The representation invariance score is
`1 / (1 + population_prediction_SD / clean_Transformer_validation_MAE)`.
One means identical predictions. It is a scale-normalized descriptive score,
not a calibrated probability and not a universal robustness metric.

Token shuffle, reversal, and leave-one-token-out are model probes. They generally
do not preserve SMILES syntax or chemistry and are never interpreted as
label-preserving attacks. Attention weights are not used as explanations.

## Results

### Predictive controls (observed)

| Model | Validation MAE (eV) | R2 |
|---|---:|---:|
| Mean predictor | 1.2022 | -0.0001 |
| Length-only ridge | 1.0274 | 0.1626 |
| Bag of token counts | 0.6574 | 0.6084 |
| Bag of token frequencies | 0.6520 | 0.6461 |
| RDKit descriptor ridge | 0.6564 | 0.6350 |
| Clean Transformer | 0.4862 | 0.7846 |
| Defended Transformer | 0.4974 | 0.7676 |

Training-set token length and target correlate at `r=-0.4247` (observed).
Frequency-only token composition is almost as predictive as raw counts and is
substantially better than length alone. The Transformer remains better than all
simple controls on this validation split.

### Equivalent-representation consistency (observed)

| Model | Mean prediction SD (eV) | Mean range (eV) | Mean pooled-embedding pairwise distance | Mean invariance score |
|---|---:|---:|---:|---:|
| Clean Transformer | 0.4756 | 1.6090 | 1.8854 | 0.5221 |
| Defended Transformer | 0.4134 | 1.3945 | 1.1137 | 0.5564 |

The paired defended-minus-clean score delta across 100 molecules is `+0.0343`;
the percentile bootstrap 95% interval is `[+0.0150, +0.0537]` using 2,000 paired
resamples. This interval quantifies sampling uncertainty for this selected
validation subset. It is not a hypothesis test, does not address training-run
variation, and must not be described as universal statistical significance.

The mean within-molecule correlation between randomized representation length
and prediction was `0.1388` for the clean model and `0.0112` for the defended
model. These per-molecule correlations are unstable when randomized lengths vary
little; raw values remain in the paired artifact.

### Position and occlusion probes (observed)

On a separate seeded sample of 100 molecules, mean clean-model drift was 0.5757 eV
for shuffled tokens and 0.7415 eV for reversed tokens. Defended-model values were
0.5896 eV and 0.5990 eV. Because these sequences were not required to be valid
SMILES, these values establish positional sensitivity only.

Across 895 token deletions from 25 seeded molecules, mean absolute prediction
change was 0.3069 eV for the clean model and 0.3224 eV for the defended model.
Deletion may alter syntax, graph, and target; token-level values are diagnostic
saliency observations rather than chemical attributions.

## Conclusions by evidence strength

### Observed

- Length alone contains predictive signal, but much less than token composition.
- Simple token composition and simple molecular descriptors have similar MAE.
- The clean Transformer materially outperforms every simple baseline on this split.
- Both Transformers vary strongly across graph-equivalent randomized SMILES.
- The defended checkpoint has a higher mean invariance score and lower pooled
  embedding variance, while paying a 0.0112 eV validation-MAE tradeoff.
- Neither defense nor the clean model is representation invariant.

### Supported

- The model uses information beyond sequence length and unordered token
  composition: its validation advantage and order sensitivity jointly support
  this bounded claim.
- Representation-level artifacts remain an important part of the learned
  function, because graph-equivalent encodings yield large prediction ranges.
- The historical adversarial training improved average randomization consistency
  for this checkpoint pair, but did not remove the failure mode.

### Speculative / not established

- The extra Transformer signal is actual polymer-chemistry interaction learning.
  It may instead include scaffold, syntax, split, or canonicalization shortcuts.
- Individual occlusion effects identify chemically causal substructures.
- The defense improvement generalizes to new splits, seeds, datasets, model
  initializations, or other attack families.
- The chosen invariance score threshold corresponds to scientific acceptability.

## Artifacts and reproduction

- `summary.json`: aggregate metrics and guardrails.
- `baseline_predictions.csv`: paired validation predictions for all baselines.
- `randomized_representation_predictions.csv`: raw equivalent-representation records.
- `per_molecule_invariance.csv`: prediction and embedding statistics per molecule/model.
- `permutation_sensitivity.csv`: non-chemical positional probes.
- `token_occlusion.csv`: per-token leave-one-out sensitivity.
- `reproducibility.json`: seeds, runtime, commit/dirty state, checkpoint hashes.
- `config_snapshot.json`: exact Phase 5, dataset, and model configuration.

Run from `materials-adversarial/` with:

```bash
.venv/bin/python scripts/run_representation_attribution.py \
  --output-dir results/representation_attribution_<new-run-id>
```

The runner refuses to overwrite an existing directory. Two preserved development
runs (`representation_attribution_initial` and
`representation_attribution_sequential_sample`) remain for provenance; the
canonical reported run is `results/representation_attribution/` and uses seeded
random sampling.
