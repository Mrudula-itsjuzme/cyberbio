from typing import List, Optional

from rdkit import Chem

from materials_adv.framework.interfaces import ValidityChecker, ConstraintSet, Candidate


class RDKitValidityChecker(ValidityChecker):
    """Parse + sanitise, with optional structural guards.

    The defaults preserve historical behaviour (parse and sanitise only). Adversarial
    evaluation should enable the guards, so that fragmentation and lost polymer
    attachment points are *rejected* rather than silently scored:

        RDKitValidityChecker(require_single_component=True,
                             require_same_attachment_count_as=source_smiles)
    """

    def __init__(
        self,
        require_single_component: bool = False,
        require_attachment_count: Optional[int] = None,
        require_same_attachment_count_as: Optional[str] = None,
    ):
        self.require_single_component = require_single_component
        self.require_attachment_count = require_attachment_count
        self.require_same_attachment_count_as = require_same_attachment_count_as

    def is_valid(self, candidate: Candidate) -> bool:
        if not candidate.identifier:
            return False
        mol = Chem.MolFromSmiles(candidate.identifier)
        if mol is None:
            return False

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return False

        if self.require_single_component and len(Chem.GetMolFrags(mol)) != 1:
            return False

        wildcards = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
        if self.require_attachment_count is not None and wildcards != self.require_attachment_count:
            return False
        if self.require_same_attachment_count_as is not None:
            reference = Chem.MolFromSmiles(self.require_same_attachment_count_as)
            if reference is None:
                return False
            expected = sum(1 for atom in reference.GetAtoms() if atom.GetAtomicNum() == 0)
            if wildcards != expected:
                return False
        return True


class EditDistanceConstraint(ConstraintSet):
    """Operator-application constraint.

    Counts operator applications (``Candidate.operator_edits``, falling back to the
    provenance tag count), NOT a SMILES edit distance. String distance is not a valid
    chemistry edit measure: canonicalisation inflates a single atom substitution to
    10-16 tokens -- see docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 7.
    """

    def __init__(self, max_edits: int):
        self.max_edits = max_edits

    def check_constraints(self, original: Candidate, candidate: Candidate) -> bool:
        if candidate.operator_edits is not None:
            return candidate.operator_edits <= self.max_edits
        return len(candidate.provenance) <= self.max_edits


class TanimotoSimilarityConstraint(ConstraintSet):
    """Structural Tanimoto similarity constraint enforcing S_Tanimoto(x, x') >= min_similarity."""

    def __init__(self, min_similarity: float = 0.5):
        self.min_similarity = min_similarity

    def check_constraints(self, original: Candidate, candidate: Candidate) -> bool:
        from materials_adv.domain.chemistry.plausibility import compute_tanimoto_similarity
        sim = compute_tanimoto_similarity(original.identifier, candidate.identifier)
        return sim >= self.min_similarity

