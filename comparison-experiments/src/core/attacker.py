import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from .schemas import AttackResult
from .budgets import BudgetManager

class Attacker(ABC):
    def __init__(self, name: str, family: str):
        self.name = name
        self.family = family

    @abstractmethod
    def attack(
        self,
        source_id: str,
        source_sequence: str,
        model: Any,
        validator: Any,
        budget: BudgetManager,
        rng: np.random.Generator,
        seed: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttackResult:
        """
        Execute the attack strategy against a single source sequence.

        Args:
            source_id: Identifier for the source sequence.
            source_sequence: The clean sequence.
            model: The target model adapter.
            validator: The constraints validator adapter.
            budget: Manager tracking queries, tokens, runtime, etc.
            rng: Seeded random number generator.
            metadata: Optional additional data.

        Returns:
            AttackResult strictly conforming to the shared schema.
        """
        pass
