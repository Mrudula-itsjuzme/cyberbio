import abc
from typing import Any, List, Optional

class Candidate:
    """A domain-agnostic candidate. ``identifier`` is whatever the domain uses
    (a SMILES string, a token sequence, ...); the generic layer never inspects it.

    ``operator_edits`` is the number of discrete OPERATOR APPLICATIONS that produced
    this candidate relative to the original source. It is maintained by the operator
    (via :meth:`AttackOperator.edit_cost`) and is what the edit budget is enforced on.
    It is deliberately NOT a string distance -- see docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md
    section 7.
    """

    def __init__(self, identifier: Any, provenance: List[str] = None,
                 operator_edits: Optional[int] = None):
        self.identifier = identifier
        self.provenance = provenance or []
        self.operator_edits = operator_edits

    def is_equivalent_to(self, other: 'Candidate') -> bool:
        return self.identifier == other.identifier

    def __repr__(self) -> str:
        return f"Candidate({self.identifier!r}, operator_edits={self.operator_edits})"


class CandidateIdentity(abc.ABC):
    """Domain-supplied notion of candidate identity, used for duplicate tracking.

    The generic search/evaluation layer must not know that a SMILES string can be
    re-canonicalised, so identity is injected rather than assumed.
    """

    @abc.abstractmethod
    def identity(self, candidate: Candidate) -> str:
        pass

    def is_equivalent(self, left: Candidate, right: Candidate) -> bool:
        return self.identity(left) == self.identity(right)


class StringIdentity(CandidateIdentity):
    """Raw-identifier identity. Correct for synthetic/string domains."""

    def identity(self, candidate: Candidate) -> str:
        return str(candidate.identifier)


class RepresentationAdapter(abc.ABC):
    @abc.abstractmethod
    def encode(self, candidate: Candidate) -> Any:
        pass

    @abc.abstractmethod
    def decode(self, representation: Any) -> Candidate:
        pass

class AttackOperator(abc.ABC):
    @abc.abstractmethod
    def apply(self, candidate: Candidate) -> List[Candidate]:
        pass

    def edit_cost(self, parent: Candidate, child: Candidate) -> int:
        """How many discrete operator edits one application represents.

        Default 1. Operators that replace a whole fragment in a single application
        may override this, but the graph-level structural change must then be reported
        separately -- one operator edit is not one atom edit.
        """
        return 1

    def make_child(self, parent: Candidate, identifier: Any, tag: str) -> Candidate:
        """Standard child construction that keeps the edit counter honest."""
        child = Candidate(
            identifier=identifier,
            provenance=parent.provenance + [tag],
            operator_edits=None,
        )
        base = parent.operator_edits if parent.operator_edits is not None else 0
        child.operator_edits = base + self.edit_cost(parent, child)
        return child

class ConstraintSet(abc.ABC):
    @abc.abstractmethod
    def check_constraints(self, original: Candidate, candidate: Candidate) -> bool:
        pass

class ValidityChecker(abc.ABC):
    @abc.abstractmethod
    def is_valid(self, candidate: Candidate) -> bool:
        pass

class AlwaysValidChecker(ValidityChecker):
    """Accepts every candidate.

    Only for domains whose operator already guarantees structural validity, and for
    abstraction tests. A chemistry benchmark must inject a chemistry checker -- it must
    never fall back to this silently.
    """

    def is_valid(self, candidate: Candidate) -> bool:
        return True

class Oracle(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, candidate: Candidate) -> float:
        pass

class Predictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, representation: Any) -> float:
        pass

class SearchStrategy(abc.ABC):
    @abc.abstractmethod
    def search(self, source: Candidate) -> Candidate:
        pass

class EvaluationProtocol(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, result: Candidate):
        pass
