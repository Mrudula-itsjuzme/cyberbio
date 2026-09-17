"""DEPRECATED compatibility shim.

``FunctionalGroupReplacementAttack`` never replaced functional groups or subgraphs; it
substitutes aliphatic carbons. It has been renamed to
:class:`~materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution.AliphaticCarbonSubstitutionAttack`.

This module is kept so that archival code (notably the frozen forensic audit
``scripts/audit_framework_v2_benchmark.py``) keeps working unchanged. New code must
import the renamed class. Do not add anything here.
"""

from materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution import (
    DEFAULT_MOTIFS,
    PROVENANCE_TAG,
    AliphaticCarbonSubstitutionAttack,
)

# Deprecated alias. Kept for archival imports only.
FunctionalGroupReplacementAttack = AliphaticCarbonSubstitutionAttack

__all__ = [
    "AliphaticCarbonSubstitutionAttack",
    "FunctionalGroupReplacementAttack",
    "DEFAULT_MOTIFS",
    "PROVENANCE_TAG",
]
