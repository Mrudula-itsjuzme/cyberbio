# Polymer-equivalence round-trip audit

RDKit parses both `*` and `[*]` as atomic-number-zero dummy atoms and serializes them as bare `*`. The frozen vocabulary, however, contains `[*]` and not bare `*`. Thus the Phase 5 serializer preserved the graph but changed the lexical attachment spelling into an unknown token.

Every original source had zero unknown tokens. Every one of the 2,907 raw variants had two unknown bare-star tokens, mapped by the historical wrapper to ID 0. Normalizing only bare dummy tokens back to `[*]` preserved canonical graph, attachment signatures, formula, charge and stereochemistry and reduced unknown counts to zero. The corrected same-bank audit still shows large drift, so the headline failure is not explained away by the lexical round-trip bug.

Original-to-canonical reserialization was also measured separately. Median drift was zero, P90 was below `5e-7 eV`, but three sources had material canonicalization drift and the maximum was `1.4361 eV`; this is reported as a distinct pipeline-induced control rather than pooled with the strict bank.
