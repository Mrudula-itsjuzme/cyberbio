# Project Overview — Audited Current State

> **Current evidence rule:** distinguish **measured result**, **interpretation**, and **hypothesis**. Validation-only results must not be presented as sealed-test findings. Chemistry-changing prediction drift must not be called physical prediction error without an independent physical oracle.

## 1. What this project is now

CyberBio is a scientific-ML robustness study centered on a harder question than "can we attack a Transformer?":

> **When a polymer-property predictor changes its output, did the chemistry change, did only the digital representation change, or did the evaluation protocol itself create a false robustness conclusion?**

The active materials lineage uses polyVERSE polymer-chain bandgap data in eV. The project began with sequence Transformers, discovered substantial equivalent-SMILES serialization dependence, audited and rejected several misleading experimental paths, moved to a graph-native representation control, stress-tested bounded chemistry changes, and formalized the point at which ML evidence must hand off to independent physical validation.

## 2. Scientific taxonomy

### Representation-preserving changes

Equivalent SMILES encode the same molecular graph. After graph identity is verified, the physical target may be inherited:

[
mathrm{mol}(x')=mathrm{mol}(x)Rightarrow y(x')=y(x).
]

These perturbations test **representation invariance**.

### Chemistry-changing changes

Substitution, insertion, deletion, and related edits can alter the molecular graph:

[
mathrm{mol}(x')
eqmathrm{mol}(x).
]

The source target is therefore not a valid candidate label. New chemistry-changing banks use:

- `source_target`
- `candidate_target: null`
- `label_basis: candidate_target_requires_independent_physical_oracle`

A large model response is **prediction drift**, not proven physical error.

## 3. Current research arc

```text
representation dependence
        ↓
equivalent-SMILES invariance audit
        ↓
data/schema + evaluation integrity audit
        ↓
training-time and inference-time remedies
        ↓
architecture-level invariance control
        ↓
bounded chemistry-changing stress
        ↓
model drift ≠ physical error
        ↓
independent physical-oracle requirement
```

This is a better description of the project than "adversarially training a Transformer."

## 4. September 21 validation-only representation review

A full validation-only experiment used **631 molecules**, requested **16 unique equivalent-SMILES views per molecule**, and obtained **15.97 views on average**. Symmetric molecules honestly expose fewer unique serializations.

No sealed test sample was opened. No model was retrained.

Three existing checkpoints were evaluated:

- frozen ordinary Transformer;
- augmentation-trained Transformer;
- GraphMPNN representation control.

### Main result

| Model | Canonical validation MAE | Mean-ensemble validation MAE | ΔMAE |
|---|---:|---:|---:|
| Ordinary Transformer | 0.5186 eV | 0.5604 eV | +0.0418 eV |
| Augmented Transformer | 0.5009 eV | 0.4844 eV | -0.0165 eV |
| GraphMPNN | 0.4112 eV | 0.4112 eV | ≈0 |

Paired molecule-bootstrap intervals from 2,000 resamples:

- ordinary Transformer: **+0.0022 to +0.0832 eV**;
- augmented Transformer: **-0.0274 to -0.0060 eV**;
- GraphMPNN: approximately **-5.9e-9 to +1.0e-8 eV**.

### Measured result

Multi-view mean inference **does not rescue the frozen ordinary Transformer**. It worsens validation MAE.

It modestly improves the existing augmentation-trained Transformer.

It is irrelevant to GraphMPNN beyond floating-point noise.

### Interpretation

Inference-time averaging is not a universal remedy for representation sensitivity. Its usefulness appears conditional on representation behavior already learned during training.

> A model cannot necessarily average its way out of a representation dependence that was learned during training.

This is an interpretation, not a demonstrated mechanism.

### Hypothesis

Training-time representation exposure and inference-time marginalization may interact. This requires independent training seeds before being generalized.

## 5. Representation sensitivity and disagreement

Ordinary Transformer equivalent-view drift:

- mean: **0.6051 eV**;
- p90: **1.0156 eV**;
- maximum observed: **4.0768 eV**.

GraphMPNN equivalent-SMILES drift:

- mean: approximately **4.8e-8 eV**;
- maximum: approximately **9.5e-7 eV**.

GraphMPNN is therefore **effectively serialization invariant up to numerical precision**. This does not imply universal robustness.

For the ordinary Transformer, prediction disagreement is associated with higher error:

- Pearson(prediction SD, absolute error): **0.291**;
- Spearman: **0.241**;
- low-disagreement quartile mean error: **0.474 eV**;
- high-disagreement quartile mean error: **0.774 eV**;
- error >0.5 eV: **34.8% vs 61.4%**.

Equivalent-view disagreement is a useful diagnostic signal under this validation protocol, but it is **not yet a calibrated uncertainty estimator** and is not causal evidence.

## 6. Audit corrections

The September 21 scientific review fixed or guarded against:

- ambiguous candidate/source target semantics;
- split seeds that were configured but ignored;
- invalid graph parsing that could silently substitute another molecule;
- incomplete all-pairs split-equivalence checking;
- pooled-candidate statistics that ignored molecule-level dependence;
- scaler lineage mistakes;
- checkpoint/vocabulary incompatibility;
- out-of-vocabulary chemistry;
- invalid equivalent-view identity;
- non-deterministic generation;
- candidate-label inheritance for chemistry-changing edits.

The current scientific-review test state is:

```text
448 passed, 1 skipped, 1 warning
```

Older `401 passed, 17 failed, 11 errors` and `429 passed` snapshots are historical, superseded repository states.

## 7. Search terminology

The current stochastic implementation is `MetropolisSearch`.

It is **not formal Metropolis-Hastings** because it does not apply the asymmetric proposal-ratio correction (q(x|y)/q(y|x)). Documentation must use "Metropolis-style" or `MetropolisSearch`, not claim a correct MH sampler.

## 8. Physical-oracle preparation

A deterministic selection pass used **68,886 Phase 12B trajectory records** and prepared:

- 20 unique candidate structures;
- 13 source structures;
- at most two candidates per source;
- fingerprint similarities from **0.156 to 0.677**;
- canonical deduplication and source balancing.

No physical calculation was performed.

RDKit validity and fingerprint similarity are screening criteria, not physical validation.

## 9. Current limitations

- new results are validation-only;
- one existing trained checkpoint per architecture was evaluated;
- training-seed variance is unresolved;
- original training provenance for the augmented Transformer and GraphMPNN is incomplete;
- equivalent-view sampling is finite;
- descriptor correlations are exploratory and not corrected for multiple comparisons;
- GraphMPNN is an architecture-level invariance control, not proof of the definitive best polymer predictor;
- independent physical validation remains external.

## 10. Next publication-grade ML experiment

Before opening the sealed test split:

1. freeze a predetermined multi-seed protocol;
2. retrain ordinary Transformer, augmented Transformer, and GraphMPNN;
3. persist checkpoint, tokenizer/vocabulary, scaler, config, dataset hash, split hashes, seed, git SHA, environment versions, and hardware metadata;
4. repeat the exact validation-only multi-view protocol;
5. separate molecule-level variation from training-seed variation;
6. freeze all analysis decisions;
7. perform exactly one final sealed-test evaluation with no tuning afterward.

## 11. Canonical reading

For the full experiment, statistics, evidence levels, reproduction commands, and do-not-claim list, see:

- [REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md](REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md)
- [CANONICAL_PROJECT_STATE.md](CANONICAL_PROJECT_STATE.md)
- [CLAIMS_LEDGER.md](CLAIMS_LEDGER.md)
