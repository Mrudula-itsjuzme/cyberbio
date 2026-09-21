# Scientific Claims Ledger

This ledger tracks the hypotheses generated during the project and their final evidentiary status based on rigorous audits.

| Claim | Status | Evidence | Limitations | Canonical Source |
| :--- | :--- | :--- | :--- | :--- |
| Sequence models natively learn chemical equivalence | **REJECTED** | Ordinary Transformer drifted ~0.614 eV on Equivalent-SMILES edits. | - | Phase 1 |
| Branch-specialization fixes equivalent-SMILES drift | **REJECTED** | Two-Branch control drifted identically to the ordinary baseline (~0.380 eV). | - | Phase 2 |
| Post-hoc representation adversarial training yields robustness | **REJECTED** | Multi-phase adversarial retraining caused catastrophic representation collapse (MAE exploded to ~2.0 eV). | - | Phase 9, 10 |
| GNN representations eliminate equivalent-SMILES drift | **SUPPORTED** | GraphMPNN equivalent-SMILES drift is strictly zero by structural definition. | - | Phase 11C, 12 |
| Chemistry-changing edits generate bounded prediction drift | **SUPPORTED** | A ≤3 edit budget capped maximum response to ~3.19 eV. | Local optima may exist. | Phase 12B |
| Original 5.961 eV drift is a true adversarial response | **REJECTED** | Derived from unbounded edit budget inflation causing non-physical fragmentation. | - | Phase 12 |
| Phase 11B GNN drift numbers were accurate | **REJECTED** | Missing padding_mask evaluation bug artificially inflated Transformer MAE to ~1.85 eV. | - | Phase 11B/C |
| Prediction drift on chemistry changes equates to absolute error | **BLOCKED** | Lack of an independent DFT oracle means the true target property response is unknown. | Blocked by environment limits. | Phase 14 |
| Target dataset generated via HSE06/PBE | **NOT_TESTED** | Speculative literature assumption. Exact metadata for `bandgap_chain.csv` is absent. | Must calibrate empirical oracle. | Phase 13A |


---

## September 21, 2026 Representation-Robustness Audit

The following claims supersede broader historical statements when discussing equivalent-SMILES multi-view inference.

| Claim | Status | Evidence | Limitation |
| :--- | :--- | :--- | :--- |
| Multi-view inference rescues the frozen ordinary Transformer | **REJECTED** | Validation MAE worsened from 0.5186 to 0.5604 eV; paired molecule-bootstrap ΔMAE interval: +0.0022 to +0.0832 eV. | One existing trained checkpoint; validation-only. |
| Multi-view inference modestly helps the existing augmentation-trained Transformer | **SUPPORTED, VALIDATION-ONLY** | Validation MAE improved from 0.5009 to 0.4844 eV; ΔMAE interval: -0.0274 to -0.0060 eV. | One existing trained checkpoint; training-seed variance unresolved. |
| GraphMPNN is effectively invariant to equivalent-SMILES serialization | **SUPPORTED UNDER THIS IMPLEMENTATION** | Mean drift ≈4.8e-8 eV; maximum ≈9.5e-7 eV; multi-view MAE unchanged to numerical precision. | This is serialization invariance, not universal robustness or proof of best polymer architecture. |
| Equivalent-view disagreement is associated with elevated error risk | **SUPPORTED, DESCRIPTIVE** | Pearson 0.291, Spearman 0.241; high-disagreement quartile mean error 0.774 eV vs 0.474 eV in low-disagreement quartile. | Not causal; not yet calibrated uncertainty. |
| Equivalent-view disagreement is a calibrated epistemic uncertainty estimator | **NOT SUPPORTED** | No calibration or multi-seed evidence establishes this interpretation. | Requires dedicated calibration and seed replication. |
| Descriptor correlations explain representation sensitivity | **NOT SUPPORTED** | Length, molecular weight, and ring count correlate with lower worst-case drift at r≈-0.27 to -0.29. | Exploratory, uncorrected for multiple comparisons, non-causal. |
| Chemistry-changing candidate labels may inherit the source label | **REJECTED** | New banks use source_target plus candidate_target=null and require an independent physical oracle. | Historical ambiguous banks remain immutable legacy artifacts. |
| Molecule bootstrap resolves training-seed uncertainty | **REJECTED** | Bootstrap correctly resamples molecules for fixed checkpoints. | Independent training-seed variance remains unmeasured. |

### Interpretation versus mechanism

**Measured result:** multi-view averaging hurts the ordinary Transformer and modestly helps the existing augmented checkpoint.

**Interpretation:** inference-time marginalization is not a universal cure and appears to depend on representation behavior already learned during training.

**Hypothesis:** training-time equivalent-representation exposure and inference-time view aggregation interact. Multi-seed retraining is required before treating this as a general mechanism.

Canonical detailed source: [REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md](REPRESENTATION_ROBUSTNESS_REVIEW_20260921.md).
