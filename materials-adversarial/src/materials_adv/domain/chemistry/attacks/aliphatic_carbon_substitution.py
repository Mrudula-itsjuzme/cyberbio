"""Aliphatic-carbon substitution operator.

RENAME NOTE
-----------
This class was previously called ``FunctionalGroupReplacementAttack``. The forensic
audit (docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 10) found that name over-promises:

* the SMARTS pattern is a single query, ``[C;X4;h1,h2,h3]`` -- one aliphatic carbon;
* 3 of the 5 replacements (O, N, F) are single atoms, so most products are plain
  atom substitutions;
* the two multi-atom replacements (C#N, C(F)(F)F) rarely survive sanitisation; when
  they do, RDKit bonds the extra atoms to the attachment atom, so no disconnected
  fragment is produced (component count is 1 in every observed candidate);
* some products have *zero* element change (bond/ring rearrangements).

So this is atom-level substitution with occasional small-fragment expansion. It is
NOT functional-group/subgraph replacement. A true fragment-replacement operator is
designed, not implemented, in docs/TRUE_FRAGMENT_ATTACK_DESIGN.md.

Edit accounting: one application is one operator edit, but it can cost up to 4
chemistry-level atom edits. Both must be reported; they are not equivalent.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from rdkit import Chem

from materials_adv.framework.interfaces import AttackOperator, Candidate
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

# (SMARTS pattern, replacement SMILES). Kept byte-identical to the historical operator
# so the verified benchmark is comparable with the audited one.
DEFAULT_MOTIFS: tuple[tuple[str, str], ...] = (
    ("[C;X4;h1,h2,h3]", "O"),
    ("[C;X4;h1,h2,h3]", "N"),
    ("[C;X4;h1,h2,h3]", "F"),
    ("[C;X4;h1,h2,h3]", "C#N"),
    ("[C;X4;h1,h2,h3]", "C(F)(F)F"),
)

PROVENANCE_TAG = "aliphatic_subst"


class AliphaticCarbonSubstitutionAttack(AttackOperator):
    """`ReplaceSubstructs` of aliphatic carbons against a small replacement library."""

    def __init__(self, motifs: Optional[Sequence[tuple[str, str]]] = None, validator=None,
                 validate_children: bool = True) -> None:
        self.motifs = tuple(motifs) if motifs is not None else DEFAULT_MOTIFS
        self.validator = validator or RDKitValidityChecker()
        # See SimpleSubstitutionAttack: skip the second parse when the search layer
        # already applies the authoritative validity gate.
        self.validate_children = validate_children

    def edit_cost(self, parent: Candidate, child: Candidate) -> int:
        return 1

    def apply(self, candidate: Candidate) -> List[Candidate]:
        mol = Chem.MolFromSmiles(candidate.identifier)
        if mol is None:
            return []

        out: List[Candidate] = []
        for pattern_smarts, repl_smiles in self.motifs:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            replacement = Chem.MolFromSmiles(repl_smiles)
            if pattern is None or replacement is None:
                continue
            try:
                products = Chem.ReplaceSubstructs(mol, pattern, replacement, replaceAll=False)
            except Exception:
                continue
            for product in products:
                try:
                    Chem.SanitizeMol(product)
                except Exception:
                    continue
                smiles = Chem.MolToSmiles(product)
                if smiles == candidate.identifier:
                    continue
                child = self.make_child(candidate, smiles, PROVENANCE_TAG)
                if not self.validate_children or self.validator.is_valid(child):
                    out.append(child)

        seen, unique = set(), []
        for child in out:
            if child.identifier not in seen:
                seen.add(child.identifier)
                unique.append(child)
        return unique
