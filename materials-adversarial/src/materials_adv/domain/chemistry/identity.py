"""Chemistry implementation of the generic candidate-identity abstraction.

The generic search layer only needs "are these two candidates the same thing?".
Chemistry answers it with a canonical graph string, so that two different SMILES
writings of the same molecule are recognised as duplicates instead of being scored
twice against the query budget.
"""

from __future__ import annotations

from rdkit import Chem

from materials_adv.framework.interfaces import Candidate, CandidateIdentity


def canonical_smiles(smiles: str) -> str:
    """Canonical graph identity, or ``None`` when the string does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


class CanonicalSmilesIdentity(CandidateIdentity):
    """Identity = RDKit canonical SMILES.

    Unparseable identifiers are identified by their raw string, so they cannot
    collide with a valid molecule.
    """

    def identity(self, candidate: Candidate) -> str:
        smiles = candidate.identifier if isinstance(candidate, Candidate) else str(candidate)
        canonical = canonical_smiles(smiles)
        if canonical is None:
            return f"<unparseable>{smiles}"
        return canonical
