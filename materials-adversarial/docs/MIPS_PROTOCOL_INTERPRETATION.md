# MIPS protocol interpretation

## Directly supported protocol details

The primary MIPS paper defines repeat-and-shift invariant training (RSIT) for
P-SMILES with two star endpoints. It states that an infinite polymer is invariant
to translation of the chosen repeat-cell boundary and to repetition of the cell.
Its examples map `*CONO*` to translated `*NOCO*` and repeated `*CONOCONO*`.
Algorithm 1 samples a Bernoulli decision for repetition and then applies a random
translation. Algorithm 2 repeatedly augments a representation and retains the
sample with maximum model loss. Source: https://arxiv.org/abs/2507.20326.

Phase 5C implements the representation transformation in Algorithm 1 only. It
does not implement Algorithm 2's loss-maximizing search, train with RSIT, or
claim the paper's downstream results.

## Required implementation choices

The accessible paper text does not fully specify the translation-offset
distribution, repeat factor distribution, trial count for every experiment, or
code-level endpoint handling. This implementation therefore:

- enumerates verified backbone boundary cuts, subject to a cap of 20 variants;
- uses repeat factor two, matching the paper's explicit example;
- includes repeat-then-translation as `SEGMENTATION`;
- preserves isomeric SMILES and rejects failed sanitization;
- validates periodic closure rather than trusting string or finite-graph
  canonical equality.

## Repository-imposed differences

The local tokenizer requires `[*]`, not the paper's typographic bare `*`. The
local GraphMPNN consumes a finite dummy-atom graph and has no periodic-boundary
operator. Consequently, a periodically equivalent translation need not collapse
to an identical input for that control model. This is measured, not assumed.

## Status

`FAITHFUL_REIMPLEMENTATION`: the repeat and translation semantics are directly
supported, while sampling and representation details are independently and
conservatively specified. It is not an exact replication of MIPS evaluation or
training.
