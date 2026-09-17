"""Atom-level substitution operator.

One application re-types exactly one heavy atom, so the operator edit count and the
chemistry-aware atom edit count agree (unlike the fragment-expanding operators).

Semantics fixed for comparability with the canonical Phase 12B pipeline
(``scripts/run_phase12b_adaptive_search_audit.py``): the audited threat model is
recorded in docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 9. Options exist to make the
differences explicit rather than accidental.
"""

from __future__ import annotations

from typing import List, Sequence

from rdkit import Chem

from materials_adv.framework.interfaces import AttackOperator, Candidate
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

DEFAULT_ELEMENTS: tuple[str, ...] = ("C", "N", "O", "S", "F")


class SimpleSubstitutionAttack(AttackOperator):
    """Replace one heavy atom with an element from ``replacements``.

    Parameters
    ----------
    replacements:
        Replacement element set. Must not include the element being replaced.
    protect_attachment:
        When True, atoms directly bonded to a polymer attachment point (``[*]``) are
        left untouched. Default False reproduces the audited V2 semantics, where
        43/416 one-edit sites sat next to an attachment atom -- a documented
        threat-model difference from Phase 12B, not an accident.
    min_heavy_neighbours:
        Skip atoms with fewer heavy neighbours than this (default 0 = substitute
        everything).
    """

    def __init__(
        self,
        replacements: Sequence[str] = DEFAULT_ELEMENTS,
        *,
        protect_attachment: bool = False,
        min_heavy_neighbours: int = 0,
        validator=None,
        validate_children: bool = True,
    ) -> None:
        self.replacements = tuple(replacements)
        if not self.replacements:
            raise ValueError("replacements must not be empty")
        self.protect_attachment = protect_attachment
        self.min_heavy_neighbours = min_heavy_neighbours
        self.validator = validator or RDKitValidityChecker()
        # Every child is already sanitised here. Set False when the caller revalidates
        # through the search's validity checker, so the same molecule is not parsed and
        # sanitised twice per proposal (a benchmark-scale performance concern only).
        self.validate_children = validate_children

    def edit_cost(self, parent: Candidate, child: Candidate) -> int:
        return 1

    def apply(self, candidate: Candidate) -> List[Candidate]:
        mol = Chem.MolFromSmiles(candidate.identifier)
        if mol is None:
            return []

        wildcard_neighbours = set()
        if self.protect_attachment:
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    for neighbour in atom.GetNeighbors():
                        wildcard_neighbours.add(neighbour.GetIdx())

        out: List[Candidate] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            if atom.GetIdx() in wildcard_neighbours:
                continue
            heavy_neighbours = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
            if heavy_neighbours < self.min_heavy_neighbours:
                continue
            for replacement in self.replacements:
                if replacement == atom.GetSymbol():
                    continue
                mutated = Chem.RWMol(mol)
                mutated.ReplaceAtom(atom.GetIdx(), Chem.Atom(replacement))
                try:
                    Chem.SanitizeMol(mutated)
                except Exception:
                    continue
                smiles = Chem.MolToSmiles(mutated)
                child = self.make_child(candidate, smiles, "subst")
                if not self.validate_children or self.validator.is_valid(child):
                    out.append(child)

        return self._deduplicate(out)

    @staticmethod
    def _deduplicate(candidates: List[Candidate]) -> List[Candidate]:
        seen, unique = set(), []
        for candidate in candidates:
            if candidate.identifier not in seen:
                seen.add(candidate.identifier)
                unique.append(candidate)
        return unique
