# Equivalence Set Specification

This specification defines the equivalence sets used for the Phase 5 representation equivalence audit.

## Families of Transformations

1. **Shift / Cyclic Shift:** Repositioning of tokens that do not alter the underlying graph.
2. **Repeat:** Trivial repetition of structural motifs if applicable.
3. **Randomized Atom Ordering:** Reordering the atom indices before generating the SMILES string. This ensures the output string is syntactically different but maps to the identical chemical graph.

## Verification
All generated representations MUST be verified by converting them back to a canonical SMILES representation using RDKit. Any variant that does not match the canonical SMILES of the source polymer is strictly excluded.

## Target Cohort
- Sample: 100-200 polymers from the test set.
- Variants per polymer: 20-50 verified chemically equivalent encodings.
