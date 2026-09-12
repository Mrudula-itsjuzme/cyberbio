"""Deprecated compatibility alias for the maintained rearrangement attack.

No checked-in experiment imports this module. Its duplicate implementation was
removed during architecture consolidation, while the historical registry name
and constructor spelling remain available for external callers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .rearrangement import RearrangementAttack
from .registry import register_attack


@register_attack("reordering")
class ReorderingAttack(RearrangementAttack):
    """Backward-compatible, deprecated spelling of ``RearrangementAttack``."""

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        window: int = 3,
        attack_budget: int = 1,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "ReorderingAttack is deprecated; use RearrangementAttack(window_size=...).",
            DeprecationWarning,
            stacklevel=2,
        )
        if attack_budget != 1:
            raise ValueError(
                "deprecated ReorderingAttack only supports attack_budget=1; "
                "use RearrangementAttack for the maintained contract"
            )
        super().__init__(rng, window_size=window, **kwargs)
