"""Graph-level structural change metrics for chemistry candidates.

Why not SMILES edit distance
----------------------------
Every chemistry candidate is canonicalised (``Chem.MolToSmiles``) before it is stored,
which reorders atoms and rewrites aromatic rings. A *single* atom substitution therefore
scores 10-16 on token edit distance (measured maximum 44). A string-based edit budget is
meaningless here: it would report ~56% of V2 candidates as edit-creep violations when the
substitution operator respects its budget exactly.

Element counts and atom/bond counts are invariant under canonicalisation, so those are
what we measure. The authoritative independent measurement used by the forensic audit
lives in ``scripts/audit_framework_v2_benchmark.py`` (frozen); this module is the live
library version used by the verified benchmark. Both use the same definitions.

See docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 7.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class GraphChange:
    element_change_count: int      # atoms re-typed or added/removed
    atom_count_delta: int
    bond_count_delta: int
    component_count_delta: int
    wildcard_count_delta: int

    @property
    def atom_edit_count(self) -> int:
        """Chemistry-aware edit count for substitution-type operators."""
        return self.element_change_count + abs(self.atom_count_delta)

    @property
    def structural_change_proxy(self) -> int:
        """Graph-level change proxy. NOT comparable to an operator-application count."""
        return (
            self.element_change_count
            + abs(self.atom_count_delta)
            + abs(self.bond_count_delta)
            + abs(self.component_count_delta)
        )

    def as_dict(self) -> dict:
        return {
            "element_change_count": self.element_change_count,
            "atom_count_delta": self.atom_count_delta,
            "bond_count_delta": self.bond_count_delta,
            "component_count_delta": self.component_count_delta,
            "wildcard_count_delta": self.wildcard_count_delta,
            "atom_edit_count": self.atom_edit_count,
            "structural_change_proxy": self.structural_change_proxy,
        }


def _element_histogram(mol) -> Counter:
    return Counter(atom.GetAtomicNum() for atom in mol.GetAtoms())


def graph_change(source_smiles: str, candidate_smiles: str) -> GraphChange:
    """Structural difference between two SMILES, invariant under canonicalisation."""
    source = Chem.MolFromSmiles(source_smiles)
    candidate = Chem.MolFromSmiles(candidate_smiles)
    if source is None or candidate is None:
        raise ValueError("graph_change requires both SMILES to parse")

    c1, c2 = _element_histogram(source), _element_histogram(candidate)
    element_changes = sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in set(c1) | set(c2)) // 2

    w1 = sum(1 for a in source.GetAtoms() if a.GetAtomicNum() == 0)
    w2 = sum(1 for a in candidate.GetAtoms() if a.GetAtomicNum() == 0)

    return GraphChange(
        element_change_count=int(element_changes),
        atom_count_delta=candidate.GetNumAtoms() - source.GetNumAtoms(),
        bond_count_delta=candidate.GetNumBonds() - source.GetNumBonds(),
        component_count_delta=len(Chem.GetMolFrags(candidate)) - len(Chem.GetMolFrags(source)),
        wildcard_count_delta=w2 - w1,
    )


def count_wildcards(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
