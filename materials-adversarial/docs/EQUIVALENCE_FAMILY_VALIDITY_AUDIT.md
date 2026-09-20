# Equivalence-family validity audit

All four generators only change RDKit atom numbering and/or the SMILES traversal root. They do not edit the parsed molecule. Across all 2,907 pairs, atom and bond counts, element multiset, formula, total formal charge, connected-component count, canonical isomeric graph, index-free stereochemistry, Morgan fingerprint, and attachment signatures match.

The polymer representation uses two unlabeled dummy atoms. Their degree, neighbor element and attachment-bond order are unchanged. Because the ends are unlabeled, serialization direction is not a distinct polymerization direction in this dataset. This conclusion would not extend to labeled/head-tail attachment conventions.

| Family | Mathematical operation | A-H audit | Status |
|---|---|---|---|
| start-point shift | choose a different SMILES traversal root on the same graph | graph, attachment atoms/bonds, repeat connectivity, unlabeled direction, stereo, charge and model atom/bond features unchanged | `VERIFIED_EQUIVALENT` |
| randomized atom ordering | graph-vertex permutation followed by noncanonical serialization | all A-H unchanged | `VERIFIED_EQUIVALENT` |
| branch-order permutation | vertex permutation plus rooted traversal changes branch emission order | all A-H unchanged | `VERIFIED_EQUIVALENT` |
| orientation reversal | reverse the internal atom index order before serialization | all A-H unchanged; it is not a chemical inversion | `VERIFIED_EQUIVALENT` |

Important correction: generic rooted serialization is not a faithful implementation of the MIPS repeat-unit shift protocol. Therefore the Phase 5B MIPS subset is empty and explicitly marked as an internal non-replication.
