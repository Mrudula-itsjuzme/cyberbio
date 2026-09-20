# Representation Equivalence Positioning

This document distinguishes the Phase 5 representation equivalence robustness audit from prior related evaluations. 

## Distinctions

1. **vs. MIPS Repeat/Shift Invariance:** Prior evaluations (e.g., MIPS) often checked for simple string-level invariants (such as cyclic shifts or repeated tokens). Our representation equivalence audit evaluates robust chemical equivalence.
2. **vs. Random SMILES Augmentation:** Previous data augmentation strategies employed random SMILES representations during training to improve generalization. In contrast, this audit does not train or alter the model. It strictly interrogates the frozen model's sensitivity to verified chemically equivalent representations (e.g., randomized atom orderings that map to the identical canonical SMILES).

## Goal
The audit tests if the model's predictive disagreement on strictly equivalent chemical representations correlates with its ordinary predictive error, thereby providing a reliability signal.
