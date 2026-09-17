"""Scaffold-preserving attack.

The scaffold is defined explicitly as the **Murcko scaffold of the source molecule**
(ring systems + linkers, with terminal attachment points stripped by RDKit). A candidate
is accepted only if that scaffold is retained as a **subgraph** of the candidate.

Retention as a subgraph is the testable property:

* modifying a scaffold atom changes its element -> the element query no longer matches;
* deleting scaffold atoms or bonds -> the subgraph match fails;
* peripheral growth or replacement off the scaffold -> the match still succeeds;
* aromatic scaffold rings must stay aromatic, because the query carries the bond orders.

The previous implementation compared a caller-supplied scaffold *string* for exact
equality, which is both easy to misconfigure and unable to distinguish "scaffold kept"
from "scaffold re-written". Status remains IMPLEMENTED_NOT_EVALUATED until the verified
benchmark runs (docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 19).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from materials_adv.framework.interfaces import AttackOperator, Candidate
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

PROVENANCE_TAG = "scaffold_preserving"


@dataclass(frozen=True)
class ScaffoldCheck:
    retained: bool
    reason: str
    scaffold_smiles: str


class ScaffoldPolicy:
    """Explicit, reusable definition of "the scaffold was preserved"."""

    def __init__(self) -> None:
        self._cache: Dict[str, Optional[object]] = {}

    def scaffold_smiles(self, smiles: str) -> str:
        """Murcko scaffold SMILES, or ``""`` for an acyclic molecule."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaffold)

    def _scaffold_query(self, scaffold_smiles: str):
        if scaffold_smiles not in self._cache:
            self._cache[scaffold_smiles] = Chem.MolFromSmiles(scaffold_smiles)
        return self._cache[scaffold_smiles]

    def check(self, scaffold_smiles: str, candidate_smiles: str) -> ScaffoldCheck:
        if not scaffold_smiles:
            return ScaffoldCheck(False, "no_scaffold_defined", "")
        candidate = Chem.MolFromSmiles(candidate_smiles)
        if candidate is None:
            return ScaffoldCheck(False, "candidate_unparseable", scaffold_smiles)
        query = self._scaffold_query(scaffold_smiles)
        if query is None:
            return ScaffoldCheck(False, "scaffold_unparseable", scaffold_smiles)
        if candidate.HasSubstructMatch(query):
            return ScaffoldCheck(True, "scaffold_retained", scaffold_smiles)
        return ScaffoldCheck(False, "scaffold_not_retained", scaffold_smiles)

    def retains(self, source_smiles: str, candidate_smiles: str) -> bool:
        return self.check(self.scaffold_smiles(source_smiles), candidate_smiles).retained


class ScaffoldPreservingAttack(AttackOperator):
    """Wrap a base operator and keep only scaffold-retaining proposals."""

    def __init__(
        self,
        base_operator: AttackOperator,
        core_scaffold_smiles: Optional[str] = None,
        policy: Optional[ScaffoldPolicy] = None,
        validator=None,
    ) -> None:
        self.base_operator = base_operator
        self.core_scaffold_smiles = core_scaffold_smiles
        self.policy = policy or ScaffoldPolicy()
        self.validator = validator or RDKitValidityChecker()
        self.rejection_reasons: Dict[str, int] = {}

    def edit_cost(self, parent: Candidate, child: Candidate) -> int:
        # The wrapper adds no edits of its own; it only filters.
        return self.base_operator.edit_cost(parent, child)

    def resolve_scaffold(self, source_smiles: str) -> str:
        return self.core_scaffold_smiles or self.policy.scaffold_smiles(source_smiles)

    def apply(self, candidate: Candidate) -> List[Candidate]:
        scaffold = self.core_scaffold_smiles or self.policy.scaffold_smiles(candidate.identifier)
        if not scaffold:
            self._reject("source_has_no_scaffold")
            return []

        out: List[Candidate] = []
        for child in self.base_operator.apply(candidate):
            if not self.validator.is_valid(child):
                self._reject("invalid_candidate")
                continue
            check = self.policy.check(scaffold, child.identifier)
            if not check.retained:
                self._reject(check.reason)
                continue
            child.provenance.append(PROVENANCE_TAG)
            out.append(child)
        return out

    def _reject(self, reason: str) -> None:
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
