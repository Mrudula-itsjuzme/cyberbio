"""Query and edit budgets.

Two independent budgets, deliberately kept separate:

``max_queries``
    How many CANDIDATE model evaluations the search may spend. The source is scored
    once before the search starts and is not charged (see
    :class:`materials_adv.framework.accounting.AttackEvaluator`). This convention is
    what makes "predictor calls == Q" a checkable claim.

``max_edits``
    How many OPERATOR APPLICATIONS may separate a candidate from the original source.
    Enforced on :attr:`Candidate.operator_edits`, which operators maintain. It is not
    a SMILES edit distance: canonicalisation makes string distance meaningless for
    chemistry candidates (audit section 7).
"""

from __future__ import annotations

from materials_adv.framework.interfaces import Candidate


class Budget:
    def __init__(self, max_queries: int, max_edits: int):
        if max_queries < 0 or max_edits < 0:
            raise ValueError("budgets must be non-negative")
        self.max_queries = max_queries
        self.max_edits = max_edits
        self.queries_used = 0

    # -- query budget ------------------------------------------------------
    def use_query(self) -> bool:
        if self.queries_used >= self.max_queries:
            return False
        self.queries_used += 1
        return True

    def is_exhausted(self) -> bool:
        return self.queries_used >= self.max_queries

    @property
    def remaining(self) -> int:
        return max(0, self.max_queries - self.queries_used)

    # -- edit budget -------------------------------------------------------
    def operator_edits(self, candidate: Candidate) -> int:
        """Operator-application count relative to the original source."""
        if candidate.operator_edits is not None:
            return int(candidate.operator_edits)
        # Legacy fallback: candidates built before operator_edits existed tracked one
        # provenance tag per application.
        return sum(1 for tag in candidate.provenance if tag != "source")

    def admits_edits(self, candidate: Candidate) -> bool:
        return self.operator_edits(candidate) <= self.max_edits

    def validate_edit_distance(self, candidate: Candidate, source: Candidate) -> bool:
        """Deprecated alias for :meth:`admits_edits`.

        Despite the name it never measured an edit *distance*; it counts operator
        applications. Kept because existing callers depend on the name.
        """
        return self.admits_edits(candidate)
