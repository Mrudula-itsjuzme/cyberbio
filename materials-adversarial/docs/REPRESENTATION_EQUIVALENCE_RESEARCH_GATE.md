# Novelty Gate

**Decision**: STOP THIS DIRECTION.

**Reasoning**:
Because the frozen model is a GraphMPNN, it is inherently permutation-invariant to the atom ordering variations that characterize SMILES string equivalents (like shift and randomization). It is mathematically impossible to produce a representation-equivalence adversarial attack against it without altering the underlying chemical graph.

No defense, contrastive training, or data augmentation is necessary.
