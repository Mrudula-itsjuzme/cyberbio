from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "materials-adversarial"
sys.path.insert(0, str(ROOT / "src"))

from materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution import AliphaticCarbonSubstitutionAttack
from materials_adv.domain.chemistry.attacks.simple_substitution import SimpleSubstitutionAttack
from materials_adv.framework.interfaces import Candidate

from .interfaces import Proposal


class CanonicalOperatorFamily:
    def __init__(self, name, operator):
        self.name = name
        self.operator = operator

    def propose(self, source: str, current: str, rng):
        parent = Candidate(current)
        return [
            Proposal(
                candidate=str(child.identifier), family=self.name,
                edit_cost=float(child.operator_edits or 1),
                trajectory=tuple(child.provenance),
            )
            for child in self.operator.apply(parent)
        ]


def canonical_families():
    return [
        CanonicalOperatorFamily("substitution", SimpleSubstitutionAttack(validate_children=False)),
        CanonicalOperatorFamily("aliphatic_substitution", AliphaticCarbonSubstitutionAttack(validate_children=False)),
    ]
