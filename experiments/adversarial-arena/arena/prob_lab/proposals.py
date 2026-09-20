"""
Arena-local proposal families for CEM/SMC.

Uses the same canonical operators as the existing Arena attackers.
Wraps materials_adv operators into the ProposalFamily protocol
expected by CEMSearch and SMCSearch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from arena.prob_lab.interfaces import Proposal

# Ensure materials-adversarial is importable
_REPO = Path(__file__).resolve().parents[4]
_SRC = _REPO / "materials-adversarial" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class _OperatorFamily:
    """Wraps a materials_adv AttackOperator into the ProposalFamily protocol."""

    def __init__(self, name: str, operator):
        self.name = name
        self._operator = operator

    def propose(self, source: str, current: str, rng) -> Sequence[Proposal]:
        from materials_adv.framework.interfaces import Candidate
        parent = Candidate(current)
        proposals = []
        for child in self._operator.apply(parent):
            candidate_str = str(child.identifier)
            proposals.append(Proposal(
                candidate=candidate_str,
                family=self.name,
                edit_cost=float(child.operator_edits or 1),
                trajectory=tuple(child.provenance),
            ))
        return proposals


def arena_proposal_families() -> list[_OperatorFamily]:
    """Return the two canonical operator families used in Arena Phase 2."""
    from materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution import (
        AliphaticCarbonSubstitutionAttack,
    )
    from materials_adv.domain.chemistry.attacks.simple_substitution import (
        SimpleSubstitutionAttack,
    )
    return [
        _OperatorFamily("substitution", SimpleSubstitutionAttack(validate_children=False)),
        _OperatorFamily("aliphatic_substitution", AliphaticCarbonSubstitutionAttack(validate_children=False)),
    ]
