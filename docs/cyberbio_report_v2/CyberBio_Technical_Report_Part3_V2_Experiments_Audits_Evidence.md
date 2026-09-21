---
title: "CyberBio Technical Report — Part III, Version 2: Experiments, Audits, Evolution, and Canonical Evidence"
author: "CyberBio Project Documentation"
date: "September 2026"
---

> **Evidence policy.** Statements labelled **Project fact** are grounded in the current `Mrudula-itsjuzme/cyberbio` repository. Sections labelled **Mathematical background** explain standard theory needed to understand the implementation. Historical results are kept only when their status is explicit: canonical, historical, rejected, blocked, or developmental-unverified.

# Scope of this volume

This volume is the evidence ledger. It answers: **what happened, in what order, which claims survived audit, which numbers are canonical, what failed, what was repaired, and what remains externally blocked?**

# 1. Repository lineages

## 1.1 Historical OpenPoly $T_g$ lineage

The earliest materials experiments used a small glass-transition-temperature dataset. This lineage produced useful lessons about sample size, shortcut learning, scaling hygiene, and invalid label inheritance, but it is archived and must not be mixed with the active bandgap pipeline.

## 1.2 Active polyVERSE bandgap lineage

The active dataset is the Ramprasad Group polyVERSE `Other/bandgap_chain.csv` lineage with 4,209 rows. The target is a 1D periodic polymer-chain electronic bandgap in eV. Exact historical DFT functional and basis/pseudopotential details are not recovered from the available local metadata.

# 2. Chronological research evolution

## Phase 1 — Establish the sequence baseline

A small Transformer regressor was used as the controlled sequence-to-property target. This provided a working predictor and an attackable discrete representation.

## Phase 2 — Expose representation sensitivity

Equivalent/Random-SMILES experiments showed that the same molecular graph could produce different Transformer outputs. This reframed the project from generic “adversarial strings” to a specific invariance problem.

## Phase 3 — Test latent semantic specialization

A two-branch Transformer attempted to learn one representation-invariant latent path and one chemistry-sensitive path. Frozen-bank evaluation did not demonstrate the intended semantic specialization. The branches could satisfy the loss through variance polarization rather than interpretable disentanglement.

## Phase 4 — Move robustness supervision to outputs

The project tested output-level consistency and teacher-style regularization. These experiments are historically informative, but chemistry-changing labels had to be reclassified carefully because original-label inheritance is physically unjustified.

## Phases 5–10 — Attribution, transfer, shortcut, and defense audits

The project accumulated evidence on representation attribution, sequence-length shortcuts, frozen candidate banks, cross-attack transfer, and robustness objectives. Some post-hoc adversarial fine-tuning phases exhibited collapse rather than a durable invariance solution.

## Phase 11B — Architecture comparison bug

An architecture comparison initially made the Transformer look far worse because evaluation omitted the padding mask. Mean pooling therefore included corrupted padded states. This result was rejected.

## Phase 11C — Canonical architecture verification

The padding path was repaired, historical checkpoints were re-verified, and the architecture frontier was recomputed under one consistent evaluator. GraphMPNN emerged as the canonical predictor.

## Phase 12 — Graph-native chemistry stress and a second bug

GraphMPNN was stress-tested under chemistry-changing search. An adaptive process accumulated edits relative to previous states, not strictly relative to the original source, producing an invalid 5.961 eV maximum.

## Phase 12B — Bounded-search repair

The search was rebuilt with a strict $\le3$ edit boundary from the original source. The repaired maximum became approximately 3.19 eV under the stated Phase 12B threat model.

## Phases 13–14 — Physical oracle design and feasibility

The project formalized the missing ground truth for chemistry-changing candidates, traced dataset provenance, built calibration manifests, and established that local quantum-chemistry execution was unavailable.

## HPC oracle package — external handoff

A backend-neutral SLURM workflow, QE/VASP scaffolding, manifests, ingestion, and analysis were implemented. Real execution remains outside the current environment.

## Framework V2 — forensic failure, then verified rerun

A developmental generic-attack benchmark was found to use a hash stub, contaminated source split, and invalid query accounting. Those numbers were rejected. The pipeline was repaired with a real hash-validated GraphMPNN wrapper and strict parsing. `verified_run_1` is the only Framework V2 result set that may be cited.

# 3. Canonical architecture table

The repository’s authoritative Phase 11C/12B table is:

### 3.1 Clean model quality and status

| Model | Parameters | Validation MAE | RMSE | $R^2$ | Status |
|---|---:|---:|---:|---:|---|
| Transformer Ordinary | 85,761 | 0.486 | 0.667 | 0.785 | canonical sequence baseline |
| Transformer ArchControl | 90,049 | 0.461 | 0.642 | 0.800 | historical |
| Transformer MixedRobust | 90,049 | 0.441 | 0.612 | 0.819 | historical |
| Transformer Augmented | 90,049 | 0.501 | 0.696 | 0.766 | historical |
| **GraphMPNN Small** | **27,585** | **0.411** | **0.595** | **0.829** | **canonical predictor** |

### 3.2 Stress/invariance metrics from the same canonical table

| Model | Equivalent-SMILES drift (eV) | Substitution drift (eV) | Deletion drift (eV) |
|---|---:|---:|---:|
| Transformer Ordinary | 0.614 | 0.291 | 0.433 |
| Transformer ArchControl | 0.382 | 0.281 | 0.362 |
| Transformer MixedRobust | 0.308 | 0.269 | 0.289 |
| Transformer Augmented | 0.147 | 0.281 | 0.169 |
| **GraphMPNN Small** | **~0** | **0.365** | **0.217** |

GraphMPNN equivalent-SMILES drift is reported as approximately $4.8\times10^{-8}$ eV from floating-point numerical effects.

# 4. What Phase 11C actually proved

Phase 11C did three things simultaneously:

1. recovered the expected historical Transformer metrics after fixing the padding path;
2. compared the sequence and graph architectures under a consistent evaluator;
3. showed that GraphMPNN achieved structural serialization invariance while also improving clean validation MAE with far fewer parameters.

The project therefore changed canonical predictor, not just defense hyperparameters.

# 5. Phase 12B bounded-search evidence

Under the Phase 12B protocol:

- sources were fixed validation molecules;
- search budgets were $Q\in\{10,20,50\}$;
- seeds were 42, 1337, and 7;
- methods were random, greedy, and Metropolis-style;
- every candidate had to remain within at most three token edits of the original source;
- cross-model transfer was measured outside the search budget.

At $Q=50$, the Phase 12B report gives mean GraphMPNN drift approximately:

| Search | Mean GraphMPNN drift (eV) |
|---|---:|
| Random | 0.868 |
| Greedy | 0.967 |
| Metropolis | 0.764 |

Metropolis found the absolute repaired maximum of about 3.19 eV, while greedy had the higher mean in that experiment.

# 6. Why 5.961 eV is invalid

The old Phase 12 stateful search allowed **edit creep**. A candidate could be within a small local step of the previous state while becoming much farther than three edits from the original source. That violated the intended threat model.

Therefore:

$$
5.961\text{ eV}\quad\text{is rejected evidence.}
$$

This is not a “slightly outdated number”; it is tied to a protocol bug.

# 7. Framework V2 developmental audit

The developmental Framework V2 benchmark is also rejected evidence. The forensic audit found:

- placeholder predictor: `float(hash(smiles) % 1000)/100`;
- source split: 25 train, 5 test, 0 validation;
- 482/540 runs exceeded declared query budget;
- the budget counter understated model asks in all 540 runs.

The apparent 6–7 eV average drifts were a hash-stub artifact, not a chemical-model result.

# 8. Framework V2 verified rerun

The repaired benchmark uses:

- canonical GraphMPNN checkpoint validated against the release manifest;
- canonical scaler;
- validation-only source selection;
- strict graph parsing;
- one budgeted scoring path;
- operator fixed when comparing search strategies;
- explicit distinction between operator count and graph edit count.

The verified run reproduced the canonical clean validation metrics exactly before any attack comparison. That preflight is essential: a benchmark is meaningless if it cannot first reproduce the model it claims to attack.

The current repository stores the repaired results under `results/framework_v2/verified_run_1/` and companion tables under `docs/SEARCH_COMPARISON_TABLE.md` and `docs/ATTACK_COMPARISON_TABLE.md`.

# 9. A subtle edit-distance audit lesson

The forensic audit found that SMILES Levenshtein distance is not a reliable chemical edit count after canonicalization. A one-atom substitution can trigger a large string rewrite. Therefore:

- Phase 12B’s token-distance rule defines **that specific threat model**;
- Framework V2 uses graph-aware edit accounting for its own operator protocol;
- the two must not be merged into a single “3 edits” claim without stating the edit definition.

# 10. Clean scaler and unit audit

The Framework V2 audit verified:

- canonical scaler equals a training-split fit to numerical tolerance;
- inverse scaling is applied exactly once;
- prediction drift is computed in eV;
- no scaled-vs-unscaled subtraction is used;
- the GraphMPNN wrapper reproduces the canonical evaluator exactly.

These checks matter because a double inverse transform or unit mismatch can create impressive-looking nonsense without any attack bug.

# 11. GraphDataset strict-mode correction

Historical `GraphDataset(strict=False)` silently advanced to another dataset row if a SMILES failed parsing. That path never became a canonical attack result, but it was a latent evaluation hazard.

The repaired adversarial predictor uses `strict=True`, so malformed candidates raise and are rejected rather than being scored as a neighboring molecule.

# 12. Data leakage and split audits

The repository distinguishes several leakages:

- exact canonical-identity overlap;
- scaffold overlap under random splitting;
- target-scaler fitting leakage;
- attack-candidate overlap;
- source-split contamination in the rejected Framework V2 developmental benchmark.

The lesson is that “no exact duplicate rows” is not enough for materials ML. Structural scaffold sharing can still make evaluation substantially easier.

# 13. Shortcut learning

Sequence length was shown to correlate with bandgap and with model behavior. The project uses this as evidence of partial shortcut reliance, not proof that sequence length fully explains the model.

A one-dimensional length-only linear model was strong enough to serve as a nontrivial baseline in earlier audits, while the Transformer still substantially outperformed it. This makes shortcut analysis a diagnostic rather than a claim that the model “only learned length.”

# 14. Explainability evidence and corrections

The repository contains multiple explainability phases. Version 2 separates developmental observations from the later canonical gradient ablation. Earlier documents that claimed unusually large star-token gradients are not treated as final when the canonical benchmark reports a zero mean boundary-star gradient in its specific ablation.

Attention weights are also not treated as causal explanations. Occlusion, gradient attribution, and shortcut controls are interpreted as complementary diagnostics.

# 15. Defense evidence in proper context

Historical defense experiments showed that Random-SMILES augmentation can reduce representation drift, and chemistry-changing consistency regularization can reduce sensitivity to a perturbation family. However:

- representation invariance became structurally solved by GraphMPNN in the final architecture;
- chemistry-changing consistency is not physical supervision;
- post-hoc adversarial training also produced collapse in some phases;
- final claims therefore emphasize architecture and oracle-backed scientific validity over “defense wins.”

# 16. Bio-Cyber branch halt

The synthetic Bio-Cyber dataset was audited with simple k-mer logistic-regression baselines:

| Features | Accuracy |
|---|---:|
| 1-mer | 50.9% |
| 2-mer | 93.6% |
| 3-mer | 98.9% |

Because a trivial 3-mer linear model nearly solved the classification task, all downstream deep-learning attack/defense claims were halted. Preserving this failure is part of the project’s methodological quality control.

# 17. Physical-oracle provenance lock

The target dataset is verified as polyVERSE `Other/bandgap_chain.csv`, 4,209 rows, 1D periodic polymer-chain bandgap, computational DFT, eV. What remains unknown is the exact historical functional and basis/pseudopotential setup.

Therefore claims such as “the dataset used HSE06” or “the dataset used PBE” are not promoted as exact-dataset facts without direct evidence.

# 18. Oracle calibration logic

A candidate physical oracle must first be tested on a deterministic calibration set. Useful metrics include

$$
\mathrm{MAE},\quad \mathrm{RMSE},\quad R^2,\quad r_{Pearson},\quad \text{bias}.
$$

Only after acceptable alignment should it be used to evaluate adversarial source/candidate deltas.

The project refuses to invent a universal threshold in advance because acceptability depends on the scientific purpose and the property mismatch.

# 19. HPC implementation status

The repository marks the following as implemented: backend contracts, structure-preparation provenance, calibration/adversarial manifests, SLURM generation, backend checks, job export, result ingestion, calibration analysis, adversarial error analysis, and uncertainty bookkeeping.

Quantum ESPRESSO and VASP backends are partial because real physics inputs and external executables/licensed data are required. The blocker is therefore external compute/parameterization.

# 20. Canonical claims ledger

| Claim | Final status |
|---|---|
| Sequence models natively learn chemical equivalence | rejected |
| Branch specialization fixes equivalent-SMILES drift | rejected |
| Post-hoc representation adversarial training guarantees robustness | rejected |
| GNN representation eliminates equivalent-SMILES drift | supported |
| Bounded chemistry edits produce substantial model sensitivity | supported |
| 5.961 eV is a valid bounded result | rejected |
| Phase 11B architecture metrics were accurate | rejected |
| Chemistry-changing prediction drift equals physical error | blocked without oracle |
| Exact dataset functional/basis is known | not established |

# 21. What can and cannot be inferred from 3.19 eV

The repaired bounded maximum says:

> under that GraphMPNN, candidate generator, edit definition, validity checks, search procedure, and query regime, the model prediction could be moved by about 3.19 eV.

It does **not** say:

> the model was wrong by 3.19 eV.

The missing quantity is the true physical response of the edited polymer.

# 22. Reproducibility map

Key current evidence files include:

- `docs/CANONICAL_RESULTS_TABLE.md`
- `docs/CLAIMS_LEDGER.md`
- `docs/RELEASE_AUDIT.md`
- `docs/PHASE11C_CANONICAL_ARCHITECTURE_VERIFICATION.md`
- `docs/PHASE12B_ADAPTIVE_SEARCH_AUDIT.md`
- `docs/PHASE13_INDEPENDENT_BANDGAP_ORACLE.md`
- `docs/PHASE13A_BANDGAP_PROVENANCE_LOCK.md`
- `docs/PHASE14_ORACLE_FEASIBILITY_AND_CALIBRATION.md`
- `docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md`
- `docs/FRAMEWORK_V2_ATTACK_RESULTS.md`
- `results/framework_v2/verified_run_1/`
- `RELEASE_MANIFEST.json`
- `hpc_oracle/`

# 23. Final research position

The final project is not best summarized as “we made an adversarially trained Transformer.” It is better summarized as:

1. we exposed a serialization vulnerability in a polymer sequence predictor;
2. we tested and rejected several inadequate internal/defense fixes;
3. we solved serialization invariance structurally with a graph model;
4. we built bounded chemistry-changing searches and repaired their protocol bugs;
5. we showed that high chemistry sensitivity cannot be called adversarial error without independent physics;
6. we built the software bridge to that physical oracle and stopped at the honest external-compute boundary.


# 24. September 21 scientific-review pass: multi-view inference and integrity gates

## 24.1 Why this pass matters

This pass did not add another attack family. It asked whether the existing conclusions survive stricter scientific bookkeeping and whether equivalent-SMILES inference-time marginalization can compensate for a representation-fragile sequence predictor.

The answer to the second question is **no for the frozen ordinary Transformer**.

## 24.2 Integrity findings and repairs

The audit found:

- ambiguous historical `target` fields in chemistry-changing banks;
- a grouped split seed that did not influence tie-breaking;
- fail-open graph parsing;
- incomplete original training provenance for existing augmented-Transformer and GraphMPNN checkpoints;
- candidate-pooled statistics where molecules should be the independent sampling unit;
- incomplete historical all-pairs canonical split-overlap checking.

New chemistry-changing banks now separate `source_target` from `candidate_target`, where the latter is null until an independent physical oracle exists. Equivalent-SMILES label inheritance requires graph identity.

Graph parsing fails closed by default. Additional integrity gates cover split completeness/disjointness, canonical cross-split overlap, units/lineage, training-only scaler equality, checkpoint-vocabulary compatibility, OOV chemistry, equivalent-view identity, deterministic generation, molecule-level bootstrap logic, and chemistry-changing label semantics.

## 24.3 Validation-only protocol

- 631 validation molecules;
- 16 requested unique equivalent views per molecule;
- 15.97 realized on average;
- 2,000 molecule-level paired bootstrap resamples;
- frozen ordinary Transformer;
- existing augmentation-trained Transformer;
- existing GraphMPNN;
- canonical/original serialization baselines;
- mean, median, and 10% trimmed-mean aggregation;
- no sealed test access;
- no retraining.

## 24.4 Multi-view result

| Model | Canonical MAE | Mean-ensemble MAE | ΔMAE | Paired molecule-bootstrap interval |
|---|---:|---:|---:|---:|
| Ordinary Transformer | 0.5186 | 0.5604 | +0.0418 | +0.0022 to +0.0832 |
| Augmented Transformer | 0.5009 | 0.4844 | -0.0165 | -0.0274 to -0.0060 |
| GraphMPNN | 0.4112 | 0.4112 | ≈0 | ≈-5.9e-9 to +1.0e-8 |

All values are validation-only eV results.

**Measured result:** averaging equivalent views worsens the ordinary Transformer and modestly improves the existing augmented checkpoint.

**Interpretation:** inference-time marginalization is not a universal rescue mechanism; it appears to depend on representation behavior already learned during training.

**Hypothesis:** training-time equivalent-representation exposure interacts with inference-time view aggregation. Multi-seed retraining is required to test this.

## 24.5 Disagreement diagnostic

Ordinary Transformer equivalent-view disagreement versus absolute error:

- Pearson = 0.291;
- Spearman = 0.241;
- low-disagreement quartile mean error = 0.474 eV;
- high-disagreement quartile mean error = 0.774 eV;
- error >0.5 eV = 34.8% vs 61.4%.

This is a descriptive risk signal, not causal evidence and not yet calibrated uncertainty.

## 24.6 Representation drift

Ordinary Transformer:

- mean drift = 0.6051 eV;
- p90 = 1.0156 eV;
- max observed = 4.0768 eV.

GraphMPNN:

- mean drift ≈ 4.8e-8 eV;
- max ≈ 9.5e-7 eV.

GraphMPNN is therefore effectively invariant to equivalent-SMILES serialization up to numerical precision, not universally robust.

## 24.7 Molecule variance versus training-seed variance

The 2,000-resample paired bootstrap treats molecules as the independent observational unit. It does not estimate variability from retraining the architecture under different random seeds.

For metric (M), conceptually:

[
mathrm{Var}(M)
=
mathrm{Var}_{s}!left[mathbb{E}_{x}(Mmid s)ight]
+
mathbb{E}_{s}!left[mathrm{Var}_{x}(Mmid s)ight].
]

The current review addresses molecule-level variability for fixed trained checkpoints. Training-seed variability remains unresolved.

## 24.8 Oracle shortlist

From 68,886 Phase 12B trajectory records, the preparation pipeline selected 20 unique candidate structures from 13 sources, with at most two candidates per source and fingerprint similarity 0.156-0.677.

No physical calculation was run. This is candidate preparation, not physical validation.

## 24.9 Test status and provenance

Current scientific-review suite: **448 passed, 1 skipped, 1 warning**.

Older test snapshots are historical and superseded.

The existing augmented Transformer and GraphMPNN are evaluation-reproducible under the new protocol, but their original training is not fully publication-reproducible because complete seed/provenance manifests were not retained.

## 24.10 Publication blocker remaining on the ML side

The highest-value extension is frozen multi-seed retraining with complete provenance, followed by the same validation-only review. Only after all model/aggregation/view-count/analysis choices are frozen should the sealed test split be opened once.

Full evidence and reproduction commands: `materials-adversarial/docs/REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md`.
