# Novelty Positioning

When positioning this research for publication or defense, it is critical to categorize the novelty accurately. We do not claim to have invented GNNs or adversarial attacks. Our novelty lies in the *intersection* and *formalization* of these concepts in physical sciences.

## 1. Methodological Novelty
*   **Formal Decoupling of Perturbations:** We explicitly define and decouple representation-preserving perturbations (SMILES syntax) from chemistry-changing perturbations (substitutions/deletions). In standard computer vision, a perturbed image is still a picture of a cat. In chemistry, a perturbed molecule is a new material. Formalizing this distinction in the context of adversarial ML is a primary contribution.
*   **The Oracle Requirement:** We formally state that for physical property prediction, adversarial error under chemistry-changing edits cannot be proven without an independent physical oracle (e.g., DFT).

## 2. Experimental Novelty
*   **Bounded Adaptive Search in Property Prediction:** We apply a strict, rule-bounded adaptive search algorithm (Random, Greedy, MCMC) specifically targeting regression tasks (bandgap), demonstrating that models are highly sensitive (up to 3.19 eV drift) even when constrained to a maximum of 3 chemically valid edits.
*   **Cross-Model Transferability of Chemical Sensitivities:** We provide evidence that chemistry-changing sensitivities transfer across fundamentally different architectures (sequence vs. graph), implying these are learned mappings of the dataset rather than arbitrary gradient artifacts.

## 3. Framework Novelty
*   **Domain-Agnostic Adversarial Pipeline:** The framework cleanly separates the `RepresentationAdapter`, `AttackOperator`, `ConstraintSet`, and `Oracle`. This modularity allows the exact same adversarial search logic to be easily adapted for genomics (DNA) or proteomics by swapping the adapter and constraint sets.

## 4. Negative-Result / Diagnostic Novelty
*   **Post-Hoc Collapse in Material ML:** We document the failure of adversarial fine-tuning on sequence representations, highlighting that models can memorize adversarial examples (post-hoc collapse) without learning the underlying structural invariance.
*   **The Inflation Bug Hazard:** We expose the danger of edit creep (stateful search accumulating edits beyond the budget from the original source) in adversarial chemical search, which led to artificially inflated vulnerability metrics (the 5.961 eV invalidation in Phase 12).

## 5. Future Physical-Validation Extension
*   **Surrogate-QC Pipeline:** We present a transparent, reproducible `CALIBRATABLE_SURROGATE` protocol to map 2D topological edits back to verifiable 3D geometries for HPC clusters, explicitly addressing the missing-provenance problem in benchmark datasets.
