# Project Story: Evaluating Adversarial Vulnerabilities in Materials ML

## Overview
This document traces the evolution of our investigation into the robustness of machine learning models for materials science, specifically predicting polymer properties like electronic bandgap from SMILES strings. What began as a straightforward adversarial robustness evaluation revealed deep complexities surrounding chemical representation, model collapse, and the fundamental definition of an adversarial attack in the physical sciences.

## The Representation Vulnerability (Phase 1-8)
We started with a standard Sequence-to-Sequence (SMILES Transformer) model. Initial tests revealed a severe vulnerability: the model exhibited high prediction drift (~0.404 eV) on "Equivalent-SMILES" edits—syntactically different but chemically identical representations of the same polymer.

We attempted to resolve this by:
1. **Branch Specialization**: Hypothesizing that separating representation learning from property regression would help. It did not; equivalent-SMILES drift remained high.
2. **Adversarial Training**: We generated adversarial equivalent-SMILES strings and retrained the model to enforce invariance.

## The Collapse Diagnosis (Phase 9-10)
Adversarial retraining resulted in catastrophic model collapse. The MAE exploded to ~2.0 eV. Diagnostics in Phase 10 revealed that forcing sequence models to map diverse, disparate token strings to identical latent representations caused the embedding space to collapse. The continuous latent manifold required for smooth property regression was destroyed.

## The Graph Selection (Phase 11-12)
Realizing that sequence representation was the root cause, we migrated to a Graph Neural Network architecture (GraphMPNN).
By modeling molecules natively as graphs, the equivalent-SMILES vulnerability was structurally eliminated (drift = 0.000 eV). The GraphMPNN also achieved a superior clean MAE (~0.411 eV) with significantly fewer parameters (27k vs 90k) than the Transformer baseline. 

GraphMPNN was officially declared the **canonical** model for the remainder of the project.

## Chemistry-Changing Attacks (Phase 12B)
Having solved the representation vulnerability, we shifted to attacks that genuinely alter the chemistry (insertions, deletions, substitutions). We developed a bounded adaptive search (Metropolis-style) capped at $\le 3$ edits to prevent unbounded polymer fragmentation (a bug that previously produced non-canonical ~5.9 eV drifts).

The repaired bounded search discovered valid chemical edits that induced up to a ~3.19 eV drift in the GraphMPNN prediction.

## The Oracle Block (Phase 13-14)
The ~3.19 eV response forced a critical scientific realization: **Is this model error, or did the true physical bandgap actually change by ~3.19 eV?**
Without an independent quantum chemistry oracle (like DFT) to compute the true property of the new candidate polymer, we could not scientifically classify the drift as an adversarial failure. 

We locked the provenance of the target dataset (`bandgap_chain.csv`) to the Ramprasad Group's `polyVERSE` repository. However, the exact DFT functional and basis set used were UNKNOWN. 

An audit of the local environment revealed that no requisite quantum chemistry software (Quantum ESPRESSO, PySCF, ASE) was available. Consequently, the independent oracle was classified as `NOT_FEASIBLE_IN_CURRENT_ENVIRONMENT`.

## Conclusion
True attack $\to$ defend training for chemistry-changing edits is currently blocked by the lack of an HPC/DFT backend. We have successfully mitigated the representation vulnerability via graph architectures and defined the exact experimental bounds for physical property attacks, paving the way for future evaluation on dedicated computational clusters.
