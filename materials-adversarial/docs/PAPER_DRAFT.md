# Title Options

1. Disentangling Representation and Chemical Robustness in Polymer Property Prediction
2. Structural Invariance and Chemistry-Changing Sensitivities in Material Property Models
3. Adversarial Evaluation of Polymer Bandgap Predictors: From Serialization to Physical Oracle Validation
4. Separating Syntax from Physics: Adversarial Robustness in Graph and Sequence Polymer Models
5. Beyond Representation Drift: Bounded Adversarial Chemistry and the Need for Physical Oracle Validation

# Abstract

Machine learning models for material property prediction are susceptible to adversarial perturbations, but evaluations often conflate representation artifacts with physical chemistry modifications. We systematically investigate the adversarial robustness of polymer bandgap predictors, demonstrating that sequence-based models (Transformers) exhibit severe representation-preserving vulnerability (0.614 eV drift for equivalent SMILES strings). By transitioning to a Message Passing Neural Network (GraphMPNN), this representation drift is structurally eliminated to numerical zero, yielding a clean validation MAE of 0.411 eV. However, the GraphMPNN remains highly sensitive to bounded, chemistry-changing stress (substitutions and deletions), with an adaptive search inducing prediction drift up to 3.19 eV. Crucially, without independent physical supervision (an oracle), it is impossible to determine whether this chemistry-changing drift represents model error or accurate physical scaling. We outline a transparent surrogate-geometry protocol for quantum chemistry calibration and establish that true adversarial evaluation in materials informatics requires separating representation instability from physical chemistry response. We further extend this framework to support generic adversarial operations, evolutionary search, and LLM-guided proposals.

# 1. Introduction

Machine learning (ML) models are increasingly deployed to predict physical properties of materials, accelerating the discovery of novel polymers and compounds. These models rely on data representations such as SMILES strings or molecular graphs to ingest chemical structures. However, these representations can introduce vulnerabilities. SMILES strings, for instance, offer non-unique, sequence-based serialization of chemical graphs. This non-uniqueness exposes models to representation-preserving attacks, where semantically identical structures yield wildly varying predictions. 

This paper investigates the adversarial robustness problem in material property prediction. A critical contribution of this work is the strict separation between representation-preserving attacks (modifying the syntax without altering the underlying chemistry) and chemistry-changing perturbations (physically altering the molecule). We demonstrate how structural invariance can remove representation vulnerabilities and explain why independent physical supervision (an oracle) is mandatory to evaluate the true adversarial error of chemistry-changing perturbations.

# 2. Background and Motivation

The prediction of electronic bandgaps is a central task in polymer informatics. Traditional methods rely on expensive Density Functional Theory (DFT) calculations. ML models, particularly deep neural networks, offer rapid screening capabilities. However, the reliability of these models under adversarial stress remains poorly understood. When a model's prediction changes drastically due to a minor perturbation, it is critical to determine if the model is failing (an adversarial vulnerability) or if the underlying physics genuinely dictate a large property shift.

# 3. Problem Definition

We define the adversarial robustness problem in materials prediction along two distinct axes:
1.  **Representation Robustness:** The model's prediction must remain invariant to changes in the data representation that correspond to the exact same physical molecule (e.g., equivalent SMILES strings).
2.  **Chemical Robustness:** The model's prediction should smoothly track true physical property changes when valid, bounded chemical modifications are applied, without exhibiting unphysical discontinuities.

# 4. Dataset and Target Property

## Electronic Bandgap
The target property is the electronic bandgap, defined as the energy difference (in eV) between the highest occupied electronic state (valence band) and the lowest unoccupied state (conduction band). It determines whether a material acts as an insulator, semiconductor, or conductor. Our models predict this continuous physical property based on the 1D periodic polymer chain representation.

## Dataset Provenance
We utilize the polyVERSE bandgap-chain dataset, comprising 1D periodic chain electronic bandgaps originally computed via DFT. The exact dataset contains 4,209 usable samples. The dataset is split into training, validation, and an exposed test set (used repeatedly during framework debugging). Validation was utilized for model selection, and target scaling was applied.

*Verified limitation:* The exact DFT functional, basis set, and original 3D periodic geometries used to generate the dataset labels are unresolved and unavailable.

# 5. Threat Model

Our evaluation strictly separates two fundamental classes of perturbation:

## A. Representation-Preserving
*   **Method:** Equivalent/randomized SMILES generation.
*   **Nature:** The underlying molecular/polymer graph remains identical.
*   **Implication:** The original physical property label is perfectly reusable. Any deviation in prediction is definitively a model error.

## B. Chemistry-Changing
*   **Method:** Substitutions, deletions, and bounded multi-substitutions.
*   **Nature:** The actual chemical structure is modified.
*   **Implication:** The original physical property label is NOT reusable. The new molecule has a new physical bandgap. Prediction drift cannot be equated with prediction error without external verification.

# 6. Models

We evaluate multiple architectures designed to predict polymer bandgaps:

1.  **Ordinary Transformer (85,761 params):** A baseline Transformer encoder regression model operating on SMILES tokens. It was evaluated to establish baseline vulnerability.
2.  **Architecture Control (90,049 params):** A two-branch variant attempting to separate syntax from semantics.
3.  **Mixed-robust Transformer (90,049 params):** A transformer trained with adversarial data augmentation.
4.  **Augmented Transformer (90,049 params):** Extensive augmentation attempt.
5.  **GraphMPNN (27,585 params):** The **Canonical Model**. A Message Passing Neural Network operating directly on molecular graphs. It was selected for achieving the highest clean validation performance and structural invariance to serialization.

*(Note: Failed specialized branches, such as the initial semantic hypothesis, are documented but not claimed as successful defenses.)*

# 7. Narrative of Discovery

1. **Initial Goal**: The project began with the goal of creating a unified attack and defense framework for material sequence models.
2. **Transformer Baseline**: The ordinary transformer was established as the baseline.
3. **Representation Vulnerability**: Equivalent-SMILES generation revealed severe representation vulnerability.
4. **Two-Branch Idea**: A two-branch specialization idea was proposed to separate semantics and syntax.
5. **Specialization Negative Result**: The two-branch architecture failed to improve robustness without performance collapse.
6. **Output-level Experiments**: Output-level consistency training and adversarial fine-tuning were attempted.
7. **Adversarial Fine-Tuning Collapse**: Representation-only fine-tuning resulted in post-hoc collapse, indicating the representation was fundamentally flawed.
8. **Unseen Chemistry Stress**: Focus shifted to chemistry-changing attacks.
9. **Early Adaptive-Search Artifact**: Unconstrained edits led to massive structural bloat.
10. **Repaired Bounded Search**: The search was constrained to <=3 edits relative to the original source.
11. **Phase 11B Bug**: The Transformer comparison was flawed due to a missing padding mask.
12. **Phase 11C Repair**: The padding mask was corrected, allowing accurate evaluation.
13. **GraphMPNN Selection**: The GraphMPNN was selected as the canonical model for solving representation drift structurally.
14. **Graph Chemistry Stress**: The GraphMPNN remained sensitive to chemistry edits.
15. **Phase 12B Attacks**: Repaired bounded search revealed a 3.19 eV drift.
16. **Physical Oracle Requirement**: We realized a physical oracle was strictly necessary.
17. **Provenance Investigation**: The original DFT geometry details were missing.
18. **Surrogate HPC Work**: A transparent `CALIBRATABLE_SURROGATE` protocol was designed and prepared for an HPC handoff.
19. **General Framework V2**: The framework was abstracted for generic structured-domain application, evolutionary searches, and LLM-guided proposals.

# 8. Extended Attack Families

This framework introduces new advanced attack families and search methods:

*   **Motif Replacement Attack** (IMPLEMENTED_NOT_EVALUATED): Replaces atoms with chemically consistent functional groups from a curated library.
    *CAUTION (see `docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md`): the implementation is atom-level substitution of aliphatic carbons (`AliphaticCarbonSubstitutionAttack`); the name over-promises and must be changed before any public claim. One application can cost up to 4 independent atom/bond edits, so motif budgets are not comparable to atom-substitution budgets.*
    *Any Framework-V2 attack number (including the 6–7 eV drift spreading through earlier drafts) is `DEVELOPMENTAL_UNVERIFIED` and must not be cited.*
*   **Scaffold-Preserving Attack** (IMPLEMENTED_NOT_EVALUATED): Constrains edits to peripheral groups, preserving the defined backbone.
*   **Evolutionary Search** (IMPLEMENTED_NOT_EVALUATED): Black-box genetic algorithm targeting property drift.
*   **Targeted Objectives** (IMPLEMENTED_NOT_EVALUATED): Supports TARGET_INCREASE, TARGET_DECREASE, TARGET_VALUE.
*   **LLM-Guided Proposals** (PROTOCOL_ONLY): Uses an LLM to propose edits under strict constraint validation. The LLM does NOT judge validity or serve as an oracle.

# 9. Generalizing the Adversarial Framework Beyond Polymer SMILES

Domain generalization requires shared attack/search mechanics combined with domain-specific components: representation adapter, validity checker, constraints, operators, and oracle. While chemistry uses SMILES/graphs and DFT oracles, this framework architecture is designed to support future DNA and protein adapters (NOT IMPLEMENTED, NOT EVALUATED, FUTURE WORK). 

*(Note: Genomics and protein domains remain explicitly future work.)*

# 10. Oracle and Physical Validation Problem

Finding a 3.19 eV prediction change raises the core physical validation problem: Does the chemical edit *actually* change the physical bandgap by 3.19 eV? The target model cannot judge its own chemistry-changing adversarial error.

We define:
*   `Delta_T`: True physical property change.
*   `Delta_G`: Model prediction change.
*   `E_G`: True prediction error (requires knowing `Delta_T`).

An independent physical reference (an Oracle, such as DFT) is mandatory to calculate `Delta_T`.
Currently, exact source DFT protocols and original periodic geometries are unavailable. We constructed a transparent, reproducible `CALIBRATABLE_SURROGATE` 3D protocol mapping 2D topologies to capped oligomers (n=2,3,4) for Quantum ESPRESSO. 
*Current Status:* `BACKEND_EXECUTION_BLOCKED` due to local HPC unavailability. The surrogate results are prepared but unexecuted.

# 11. Results

(See Canonical Comparison Tables in Section 11 of the repository for full numerical details.)
The core results establish the GraphMPNN as the canonical predictor, confirming that structural graph representations eliminate serialization vulnerabilities. However, bounded chemistry modifications reliably induce significant predictive drift across all tested architectures.

# 12. Discussion

The deeper interpretation of these findings is that serialization sensitivity in materials ML is largely an architectural representation problem, not just a data problem. Structural invariance (via graphs) can entirely remove the representation attack class. However, chemistry responsiveness is fundamentally different from representation instability. True "robustness" in materials discovery cannot mean "ignoring chemistry." Therefore, physical oracle supervision is strictly necessary for evaluating and training against chemistry-changing adversarial examples.

# 13. Limitations

*   **Exposed Test Set:** The test set was queried during iterative debugging.
*   **Incomplete Provenance:** The exact DFT physics protocol and 3D geometries of the source dataset are unknown.
*   **No Ground-Truth Labels for Edits:** Real physical bandgaps for the computationally generated candidate molecules are unavailable.
*   **Unexecuted Surrogate Calibration:** The 9-job surrogate QC calibration pilot remains unexecuted.
*   **Partial Edit Space:** The bounded operators cover only a subset of valid chemistry (insertion not canonically tested).
*   **Heuristic Search:** The discrete black-box search is not exhaustive.
*   **Task Specificity:** Conclusions are specific to polyVERSE bandgap regression.

# 14. Conclusion

This project demonstrates that representation robustness and chemical robustness must be rigorously separated in material informatics. While graph-based models (GraphMPNN) successfully eliminate equivalent-SMILES dependence by mathematical construction, they expose substantial model sensitivity to bounded, valid chemical perturbations. Ultimately, determining true adversarial error for chemistry-changing attacks requires independent physical property supervision, highlighting the critical role of the external oracle in future robustness research.
