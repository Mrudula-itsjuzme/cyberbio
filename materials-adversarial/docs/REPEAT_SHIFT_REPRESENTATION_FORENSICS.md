# Repeat/shift representation forensics

## Repository representation

The evaluated `original_representation` field is PSMILES: an ordinary finite
SMILES graph containing exactly two dummy atoms, conventionally written `[*]`.
Each dummy must be degree one. Its neighbor and connecting bond define one end
of the repeat unit. The record is a finite repeat cell, not a symbolic BigSMILES
expression, an explicit infinite chain, or a three-dimensional periodic cell.

Evidence in the repository:

- `src/materials_adv/data/tokenizer.py` recognizes bracket atoms before other
  tokens and therefore preserves `[*]` as one token. Tokenization and
  detokenization are exact string inverses.
- `data/processed/vocab.json` contains `[*]`; a bare `*` is not the canonical
  training spelling.
- `src/materials_adv/domain/chemistry/polymer_repeat.py` requires two wildcard
  atoms and constructs head-to-tail finite oligomers by joining their neighbors.
- `src/materials_adv/data/preprocessing.py` explicitly warns that RDKit treats
  `*` only as a dummy atom and that canonical finite-graph equality is not
  polymer identity.
- The Phase 5B cohort manifest contains the same finite two-endpoint strings.

## Endpoint and direction semantics

The file format does not label head versus tail. Atom order therefore cannot by
itself establish direction. This protocol accepts a boundary translation only
after closing the two endpoint-neighbor atoms with the endpoint bond type and
showing isomeric periodic-graph identity. It retains stereochemistry and formal
charge. A transformation is rejected when there are not exactly two degree-one
endpoints, endpoint bond types differ, closure cannot be sanitized, tokenization
introduces an unknown, or the result exceeds 256 tokens.

The repository has no canonical polymer-cell minimizer and no established rule
that arbitrary orientation reversal preserves a directional repeat. Phase 5C
therefore does not generate a separate orientation family. Repetition is an
explicit head-to-tail tiling of the supplied cell. Translation cuts a different
bond on the endpoint-to-endpoint backbone of the periodic closure; branch bonds
are never selected.

## Canonicalization and limitations

RDKit canonicalization may reorder a finite graph, but cannot prove periodic
polymer equivalence. Phase 5C uses RDKit only as the graph engine inside an
explicit periodic-closure validator. Aromatic cells for which closure or a cut
cannot be sanitized are excluded. The representation does not encode tacticity,
end groups, molecular-weight distribution, crystal packing, or a 3D cell; this
study consequently establishes 1D periodic-topology equivalence only.
