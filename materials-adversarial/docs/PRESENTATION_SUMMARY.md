# Presentation Summary: Adversarial Robustness in Polymer Property Prediction

## Objective
To systematically evaluate and improve the adversarial robustness of machine learning models predicting physical properties (bandgap) of polymeric materials.

## The Problem
Standard sequence models (Transformers) trained on SMILES strings exhibit a severe representation vulnerability: predicting wildly different physical properties for the exact same molecule depending on how its chemical graph is serialized into a string.

## Architecture Evolution & The Canonical Model
* **Failed Hypothesis**: Adversarial training on sequence models caused post-hoc collapse rather than learning chemical semantics.
* **Structural Solution**: Moving to a Message Passing Neural Network (GraphMPNN) structurally eliminated the equivalent-SMILES vulnerability by operating directly on the permutation-invariant molecular graph.
* **Canonical Model**: The GraphMPNN achieves `0.411 eV` clean validation MAE and ≈ `0 eV` equivalent-SMILES drift.

## Strongest Results
* We successfully decoupled representation-preserving vulnerabilities from chemistry-changing vulnerabilities.
* The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately `3.19 eV`. Deletion was evaluated separately as a fixed stress.
* We verified that some of these sensitivities transfer across different model architectures.

## Major Bugs Caught and Corrected
* **Phase 11B Evaluator Bug**: Transformer evaluation omitted the padding mask, corrupting metrics.
* **Phase 12 Adaptive Search Bug**: Edit creep occurred because search states were not strictly bounded to <=3 edits relative to the original source, falsely registering a `5.961 eV` drift. This was corrected in Phase 12B.

## The Physical-Oracle Limitation
* While we can induce a 3.19 eV change in prediction, we cannot mathematically prove this is a model error. The chemical edit might actually change the physical bandgap by 3.19 eV.
* Determining truth requires an independent physical oracle (Density Functional Theory via Quantum ESPRESSO).
* Because original 3D geometries were unavailable, we developed a transparent `CALIBRATABLE_SURROGATE` 3D construction protocol.

## The Future Path
* Execution of the surrogate QC calibration pilot on an HPC cluster is strictly required to ground prediction drift in physical reality.
* Upon successful calibration, the framework is designed to close the attack→defend loop, natively extending to robust adversarial training governed by real physical physics.
