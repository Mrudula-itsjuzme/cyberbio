# Canonical Repository Audit

This document summarizes the current state of the canonical `materials-adversarial/` implementation and outlines what components can be reused, what should be isolated, and what rules must be followed when building the exploratory comparison framework in `comparison-experiments/`.

## 1. Reusable Canonical Components
The following components from the canonical pipeline are mathematically rigorous and should be reused via thin read-only adapters:
- `TargetScaler` (data scaling and inverse-transformation).
- `PSmilesTokenizer` and its vocabulary encoding.
- `GraphMPNNPredictor` / `TwoBranchTransformerRegressorModel` model wrappers and checkpoints.
- `RDKitValidityChecker` and `ChemicalPlausibilityValidator` (syntactic and basic structural checks).
- `TanimotoSimilarityConstraint` and `EditDistanceConstraint`.
- Split utilities defining the canonical Train/Val/Test boundaries.
- Pre-existing edit operators (`SimpleSubstitutionAttack`, `AliphaticCarbonSubstitutionAttack`, etc.) as primitive mutations.

## 2. Components That Should NOT Be Reused Directly
- The `AttackEfficiencyBenchmark` and other canonical experiment runners: These are deeply tied to the canonical experimental protocol and narrative.
- The `ProbabilisticMCMCAttack` class should not be copied/forked. Instead, it must be wrapped in an adapter for evaluation to maintain single-source-of-truth.
- Any result serialization formatting that hardcodes canonical metric columns.

## 3. Stable Interfaces Available
- `Candidate` representing an adversarial proposal.
- `Predictor` base class.
- `ValidityChecker` and `ConstraintSet`.
- `SearchStrategy` (though we may wrap this in our own `Attacker` interface for consistent budget tracking).

## 4. Assumptions and Risks
- **Data Leakage Risk**: New attackers (like LLMs or RL) must strictly avoid accessing the sealed test set (`test_sealed: true`). Only source input sequences can be provided. Ground-truth `y` values must not be visible during the attack optimization process (unless evaluating error-maximization targets where allowed).
- **Incompatibilities**: Advanced attackers (like GANs or LLMs) generate sequences directly, whereas the canonical framework emphasizes discrete token edits. The budget accounting manager must carefully abstract "model queries" separately from "LLM tokens" or "generator forward passes" so comparisons remain fair.
- **Scientific Constraint**: Reusing the RDKit validator does not imply we claim generated sequences are physically synthesizable. They are strictly evaluated on parsing validity and constraint satisfaction per the canonical thesis limits.
