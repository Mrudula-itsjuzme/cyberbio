# Representation Robustness Review — 2026-09-21

> **Status:** validation-only scientific audit and inference-time robustness study.  
> **Sealed test split:** not opened.  
> **Retraining:** none.  
> **Independent sampling unit:** molecule.

## Evidence vocabulary

This document uses three explicit evidence levels:

- **Measured result:** directly computed from the recorded experiment.
- **Interpretation:** a conservative explanation consistent with the measurements, but not itself directly proven.
- **Hypothesis:** a mechanism or generalization requiring a new experiment.

This separation is mandatory when reusing the findings below.

## 1. Why this review was run

The review asked whether equivalent-SMILES multi-view inference can rescue representation sensitivity in existing polymer predictors without retraining, while simultaneously auditing evaluation integrity, label semantics, split leakage, graph parsing behavior, and statistical sampling units.

The strongest result is negative: **multi-view inference does not rescue the frozen ordinary Transformer.** It modestly improves the existing augmentation-trained Transformer and is numerically irrelevant for the graph-native control.

## 2. Scientific-integrity findings

The audit identified seven dangerous or misleading conditions:

1. Historical chemistry-changing substitution/deletion banks stored the source property in a generic `target` field. Existing evaluators used it for source clean-MAE calculations rather than candidate supervision, but the schema made accidental candidate-label inheritance possible.
2. `grouped_split(..., seed=...)` ignored the supplied seed, so the configuration implied a stochastic control that did not actually affect tie-breaking.
3. `GraphDataset` could silently substitute another molecule when parsing failed.
4. Existing Phase 11C augmented-Transformer and GraphMPNN checkpoints do not contain complete deterministic training provenance.
5. Some historical invariance analyses pooled candidate views rather than consistently treating source molecules as the independent sampling unit.
6. Historical Phase 11C leakage checking covered train-validation canonical overlap, not every split pair.
7. Historical chemistry-changing banks remain immutable legacy artifacts with the ambiguous `target` schema.

The stochastic search implementation is correctly named `MetropolisSearch`. It is **not** formal Metropolis-Hastings because no asymmetric proposal-ratio correction is applied.

## 3. Corrections

### 3.1 Chemistry-changing candidate semantics

New chemistry-changing banks write:

```text
source_target
candidate_target: null
label_basis: candidate_target_requires_independent_physical_oracle
```

The scientific rule is:

[
mathrm{mol}(x')
eq mathrm{mol}(x)
quadRightarrowquad
y(x')	ext{ is not inherited from }y(x).
]

Equivalent-SMILES records may inherit the property only after graph equivalence is verified:

[
mathrm{mol}(x')=mathrm{mol}(x)
quadRightarrowquad
y(x')=y(x).
]

Evaluators remain backward-compatible with immutable historical banks, but the ambiguous schema is treated as legacy rather than a template for new experiments.

### 3.2 Fail-closed behavior

The review added or required fail-closed checks for:

- complete split membership;
- disjoint split membership;
- canonical-equivalent overlap across all split pairs;
- dataset lineage and physical units;
- target-scaler equality to training-only statistics;
- checkpoint/vocabulary compatibility;
- out-of-vocabulary chemistry;
- equivalent-view graph identity;
- deterministic view generation;
- molecule-level bootstrap sampling;
- chemistry-changing label handling;
- graph invalid-input handling.

Invalid graph input now fails closed by default instead of advancing to another row.

## 4. Validation-only multi-view protocol

- Validation molecules: **631**
- Requested equivalent views per molecule: **16**
- Mean realized unique views: **15.97**
- Reason some molecules have fewer views: graph symmetry / serialization degeneracy
- Bootstrap resamples: **2,000**
- Models:
  - frozen ordinary Transformer;
  - existing augmentation-trained Transformer;
  - existing GraphMPNN.
- Aggregation:
  - arithmetic mean;
  - median;
  - 10% trimmed mean.
- Baselines:
  - canonical serialization;
  - original dataset serialization.
- Measured diagnostics:
  - validation MAE;
  - per-molecule drift;
  - prediction variance;
  - prediction range;
  - worst-case representation drift;
  - inference cost;
  - Pearson/Spearman disagreement-error association;
  - disagreement bins;
  - low/high disagreement comparisons;
  - descriptor-based failure analysis.

No sealed test sample was opened and no model was retrained.

## 5. Main quantitative result

| Model | Canonical validation MAE | Mean-ensemble validation MAE | ΔMAE = ensemble - canonical | Molecule-bootstrap interval |
|---|---:|---:|---:|---:|
| Ordinary Transformer | 0.5186 eV | 0.5604 eV | +0.0418 eV | +0.0022 to +0.0832 eV |
| Augmented Transformer | 0.5009 eV | 0.4844 eV | -0.0165 eV | -0.0274 to -0.0060 eV |
| GraphMPNN | 0.4112 eV | 0.4112 eV | approximately 0 | approximately -5.9e-9 to +1.0e-8 eV |

### Measured result

For the **ordinary Transformer**, averaging equivalent-SMILES predictions **worsened** validation MAE. The paired molecule-bootstrap interval lies entirely above zero.

For the **augmentation-trained Transformer**, equivalent-view averaging produced a modest validation improvement. The paired molecule-bootstrap interval lies entirely below zero.

For **GraphMPNN**, equivalent-view aggregation changed nothing beyond floating-point noise.

### Interpretation

Inference-time marginalization over equivalent representations is **not a universal remedy** for representation dependence. Its usefulness appears conditional on the trained predictor already exhibiting useful consistency across equivalent representations.

A concise interpretation is:

> A model cannot necessarily average its way out of a representation dependence that was learned during training.

This sentence is an **interpretation**, not a measured mechanistic fact.

### Hypothesis

Training-time equivalent-representation exposure and inference-time view marginalization may interact. Multi-seed retraining is required to determine whether the observed contrast between ordinary and augmented Transformers generalizes beyond these existing checkpoints.

## 6. Representation drift

For the ordinary Transformer:

- mean equivalent-view drift: **0.6051 eV**;
- p90 drift: **1.0156 eV**;
- maximum observed drift: **4.0768 eV**.

For GraphMPNN:

- mean equivalent-SMILES drift: approximately **4.8e-8 eV**;
- maximum observed drift: approximately **9.5e-7 eV**.

GraphMPNN is therefore described as **effectively invariant to equivalent-SMILES serialization up to numerical precision**, not as universally robust.

## 7. Representation disagreement as a diagnostic

For the ordinary Transformer:

- prediction-SD versus absolute error:
  - Pearson: **0.291**;
  - Spearman: **0.241**.
- low-disagreement quartile mean error: **0.474 eV**;
- high-disagreement quartile mean error: **0.774 eV**;
- fraction with absolute error > 0.5 eV:
  - low-disagreement quartile: **34.8%**;
  - high-disagreement quartile: **61.4%**.

### Measured result

Higher equivalent-view disagreement is associated with higher prediction error in this validation experiment.

### Interpretation

Equivalent-view disagreement contains useful information for **risk stratification** or **failure detection**.

### Not established

The current experiment does **not** establish disagreement as:

- a calibrated uncertainty estimator;
- a pure measure of epistemic uncertainty;
- a causal explanation of prediction error.

Calibration, independent training seeds, and a frozen final test evaluation are still required.

## 8. Descriptor-based failure analysis

For the ordinary Transformer, greater sequence length, molecular weight, and ring count were each associated with **lower** worst-case representation drift, with correlations around (rapprox-0.27) to (-0.29).

These are:

- exploratory;
- weak-to-moderate;
- uncorrected for multiple exploratory comparisons;
- descriptive rather than mechanistic.

They must not be presented as explanations for representation sensitivity.

## 9. Physical-oracle shortlist

A deterministic preparation pass selected candidate chemistry-changing examples from **68,886** Phase 12B trajectory records.

The shortlist contains:

- **20** unique candidate structures;
- **13** unique source structures;
- at most **2** candidates per source;
- fingerprint similarities spanning **0.156-0.677**;
- canonical deduplication;
- source balancing;
- stratified selection.

No physical calculation was performed.

RDKit validity and fingerprint similarity are **not** physical validation. Chemistry-changing candidates still require an independent physical oracle before prediction drift can be called physical prediction error.

## 10. Current test state

The scientific-review branch completed with:

```text
448 passed, 1 skipped, 1 warning
```

The older `401 passed, 17 failed, 11 errors` snapshot records a pre-review repository state and is **superseded**, not current evidence.

Historical test outputs should be preserved when useful for the audit trail, but any canonical status document must label them as superseded.

## 11. Statistical unit and unresolved seed variance

The paired bootstrap resamples **molecules**, which is the correct independent observational unit for the multi-view experiment. It quantifies molecule/sample variability for the fixed trained checkpoint.

It does **not** estimate training-seed variability.

Conceptually, for an evaluation metric (M),

[
mathrm{Var}(M)
=
mathrm{Var}_{s}!left[mathbb{E}_{x}(Mmid s)ight]
+
mathbb{E}_{s}!left[mathrm{Var}_{x}(Mmid s)ight],
]

where (s) denotes the training seed and (x) the sampled molecule.

The current molecule bootstrap informs the second component for a fixed trained model. Because only one trained checkpoint per architecture is evaluated here, the first component remains unresolved.

## 12. Training-provenance limitation

Evaluation of the existing augmented Transformer and GraphMPNN is reproducible under the review protocol. Their original training is **not fully publication-reproducible** because complete training-seed/provenance metadata was not preserved.

Therefore:

- evaluation reproducibility: supported;
- complete historical training reproducibility: not supported;
- architecture-level seed stability: not yet established.

## 13. Publication-grade next experiment

The next justified ML experiment is frozen multi-seed retraining of:

1. ordinary Transformer;
2. augmentation-trained Transformer;
3. GraphMPNN.

Five predetermined seeds are preferred; three are acceptable only if compute constraints require it.

Before any training, freeze and commit:

- seed list;
- train/validation/test split hashes;
- dataset hash;
- model/training configuration;
- tokenizer/vocabulary hash;
- target-scaler hash;
- checkpoint naming convention;
- package/software versions;
- git commit;
- hardware metadata.

For each seed persist:

- checkpoint;
- tokenizer/vocabulary;
- scaler;
- config;
- dataset/split hashes;
- metrics;
- seed;
- git SHA;
- package/environment versions;
- hardware metadata.

Then repeat the **same validation-only** multi-view protocol.

Do not open the sealed test split until architecture, aggregation method, number of views, reporting metrics, analysis protocol, and seed protocol are frozen.

After that freeze, perform **one final sealed-test evaluation with no tuning afterward**.

## 14. Claims ledger for this review

### Supported by this validation experiment

- the ordinary Transformer exhibits substantial equivalent-SMILES representation sensitivity;
- multi-view mean inference worsens the frozen ordinary Transformer under this protocol;
- multi-view mean inference modestly improves the existing augmentation-trained Transformer under this protocol;
- GraphMPNN is effectively invariant to equivalent-SMILES serialization up to numerical precision;
- equivalent-view disagreement is associated with elevated error risk;
- chemistry-changing candidate labels cannot scientifically be inherited from their sources;
- GraphMPNN is a useful representation-invariance control.

### Not yet supported

- multi-view inference universally improves Transformer robustness;
- augmentation universally solves representation dependence;
- equivalent-view disagreement is a calibrated uncertainty measure;
- descriptor correlations explain the mechanism of representation sensitivity;
- GraphMPNN is the universally best polymer predictor;
- chemistry-changing prediction drift is physical prediction error;
- the shortlisted chemistry-changing candidates are physically validated;
- the observed architecture differences are stable across independent training seeds.

### External-physics boundary

Independent physical validation of chemistry-changing candidates remains blocked on an external physical oracle protocol and compute.

## 15. Reproduction commands

From `materials-adversarial/`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests -q
```

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/run_representation_robustness_review.py \
  --n-views 16 \
  --seed 20260921 \
  --output-dir results/representation_robustness_review/20260921_validation_n16_final
```

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/select_physical_oracle_candidates.py \
  results/phase12b_adaptive_search/1789331655/attack_trajectories.jsonl \
  results/physical_oracle_preparation/20260921/candidate_shortlist.csv \
  --n 20 \
  --max-per-source 2
```

## 16. Artifact map

Primary result artifacts:

- `results/representation_robustness_review/20260921_validation_n16_final/summary.json`
- `results/representation_robustness_review/20260921_validation_n16_final/integrity_audit.json`
- `results/physical_oracle_preparation/20260921/candidate_shortlist.csv`

Core implementation associated with the review:

- `src/materials_adv/evaluation/multiview.py`
- `src/materials_adv/evaluation/integrity.py`
- `src/materials_adv/oracle/candidate_selection.py`
- `src/materials_adv/data/splits.py`
- `src/materials_adv/data/graph_dataset.py`
- `scripts/run_representation_robustness_review.py`
- `scripts/select_physical_oracle_candidates.py`
- `scripts/generate_phase3_banks.py`
- `scripts/generate_deletion_candidates.py`
- `scripts/evaluate_phase3_banks.py`
- `scripts/evaluate_phase4.py`
- `tests/test_scientific_review.py`

## 17. Research narrative after this review

CyberBio should not be summarized primarily as adversarial training of a Transformer.

The stronger research arc is:

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

The project has increasingly become a study of:

- what constitutes a scientifically valid adversarial perturbation;
- which invariances should be designed into a representation versus learned from data;
- how evaluation protocol mistakes can manufacture false robustness conclusions;
- where machine-learning evidence ends and physical validation must begin.

## 18. Do not claim

Do not claim that:

- validation-only numbers are final test performance;
- multi-view inference is a universal defense;
- augmentation is a universal cure;
- disagreement is calibrated uncertainty;
- descriptor correlations establish mechanism;
- GraphMPNN is the definitive polymer predictor;
- RDKit validity is physical validity;
- 3.19 eV prediction drift is 3.19 eV physical error;
- historical augmented-Transformer/GraphMPNN training is fully reproducible;
- the current molecule bootstrap captures training-seed variability.
