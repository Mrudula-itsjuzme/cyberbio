# FINAL RESEARCH STATUS — Audited September 21, 2026

> This page supersedes earlier completion percentages and stale repository-wide test counts. CyberBio is scientifically mature in several components, but it is **not** correct to describe the entire project as 100% implementation-complete or fully publication-reproducible.

## Current canonical position

CyberBio has evolved from a generic adversarial-training project into an audit-driven study of:

- representation dependence;
- equivalent-SMILES invariance;
- evaluation and data-integrity failure modes;
- architecture-level invariance;
- bounded chemistry-changing sensitivity;
- the distinction between model drift and physical error;
- the point at which independent physics is required.

## Current test state

Scientific-review suite:

```text
448 passed, 1 skipped, 1 warning
```

Older `401 passed, 17 failed, 11 errors` and `429 passed` records are historical snapshots and must not be cited as current repository status.

## Strongest current validation-only result

The September 21 multi-view review evaluated 631 validation molecules, up to 16 deterministic equivalent-SMILES views per molecule, and three existing checkpoints without retraining.

| Model | Canonical validation MAE | Mean-ensemble validation MAE | ΔMAE |
|---|---:|---:|---:|
| Ordinary Transformer | 0.5186 eV | 0.5604 eV | +0.0418 eV |
| Augmented Transformer | 0.5009 eV | 0.4844 eV | -0.0165 eV |
| GraphMPNN | 0.4112 eV | 0.4112 eV | ≈0 |

The ordinary Transformer result is a negative finding: multi-view inference did not rescue it.

## What is currently well supported

- equivalent-SMILES serialization can produce substantial Transformer prediction drift;
- GraphMPNN is effectively invariant to equivalent-SMILES serialization up to numerical precision under the current implementation;
- multi-view averaging worsens the frozen ordinary Transformer under the current validation protocol;
- multi-view averaging modestly improves the existing augmentation-trained Transformer under the current validation protocol;
- equivalent-view disagreement is associated with elevated prediction error risk;
- chemistry-changing candidate labels cannot scientifically inherit source labels;
- the Phase 12B bounded-search maximum is model prediction drift, not proven physical error;
- the 5.961 eV edit-creep result is rejected;
- Framework V2 developmental hash-stub results are rejected;
- Bio-Cyber deep-model claims are halted because simple k-mer baselines expose severe leakage.

## What is not yet supported

- universal benefit from multi-view inference;
- universal effectiveness of augmentation;
- calibrated epistemic uncertainty from equivalent-view disagreement;
- causal explanations from descriptor correlations;
- universal GraphMPNN superiority for polymer prediction;
- training-seed robustness across architectures;
- physical correctness of chemistry-changing shortlisted candidates;
- fully reproducible historical training of the augmented Transformer and GraphMPNN.

## Current reproducibility status

### Evaluation reproducibility

Strong for the September 21 representation-review protocol, including:

- deterministic equivalent-view generation;
- fail-closed integrity gates;
- molecule-level paired bootstrap;
- frozen validation-only evaluation;
- explicit artifact paths and commands.

### Historical training reproducibility

Incomplete for the existing augmented Transformer and GraphMPNN because complete original seed/provenance metadata was not retained.

This distinction must remain explicit.

## Physical-oracle boundary

The repository can prepare and analyze candidate sets and contains physical-oracle workflow infrastructure, but no independent physical calculation has yet supplied ground-truth bandgaps for the shortlisted chemistry-changing candidates.

Therefore:

[
	ext{prediction drift} 
eq 	ext{physical prediction error}
]

until an independent property oracle supplies the candidate response.

## Remaining publication blockers

### ML-side

Frozen multi-seed retraining with complete provenance for:

- ordinary Transformer;
- augmentation-trained Transformer;
- GraphMPNN.

Then repeat the same validation-only protocol, freeze all choices, and perform one final sealed-test evaluation with no tuning afterward.

### Physics-side

Independent physical-oracle calibration and execution for chemistry-changing candidates.

This requires external scientifically justified compute and cannot be replaced by RDKit validity or fingerprint similarity.

## Canonical detailed sources

- `materials-adversarial/docs/REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md`
- `materials-adversarial/docs/CANONICAL_PROJECT_STATE.md`
- `materials-adversarial/docs/CLAIMS_LEDGER.md`
- `docs/cyberbio_report_v2/CyberBio_Project_Overview_V2.md`
- `docs/cyberbio_report_v2/CyberBio_Technical_Report_Part3_V2_Experiments_Audits_Evidence.md`

## Do not summarize this project as

> "We adversarially trained a Transformer and made it robust."

That framing is now scientifically obsolete.

The better summary is:

> CyberBio audits how scientific ML models separate representation from chemistry, tests whether robustness fixes survive strict evaluation, and identifies the boundary where model-level evidence must yield to independent physical validation.
