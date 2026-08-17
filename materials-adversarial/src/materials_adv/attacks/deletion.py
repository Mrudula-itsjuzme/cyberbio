from typing import Sequence

import numpy as np

from materials_adv.attacks.base import BaseAttack
from materials_adv.attacks.generator import AttackOutcome
from materials_adv.attacks.registry import register_attack
from materials_adv.attacks.token_space import TokenRole, classify_token, editable_positions


@register_attack("deletion")
class DeletionAttack(BaseAttack):
    def __init__(
        self,
        rng: np.random.Generator,
        attack_budget: int = 1,
        protect_attachments: bool = True,
        protect_ring_closures: bool = True,
        protect_branches: bool = True,
    ):
        super().__init__(rng)
        self.attack_budget = attack_budget
        self.protect_attachments = protect_attachments
        self.protect_ring_closures = protect_ring_closures
        self.protect_branches = protect_branches

    def _eligible_deletion_positions(self, tokens: list[str]) -> list[int]:
        editable = editable_positions(
            tokens,
            protect_attachments=self.protect_attachments,
            protect_ring_closures=self.protect_ring_closures,
            protect_branches=self.protect_branches
        )
        
        # Deletion is further constrained: we don't delete elements unless they are atoms or bonds
        allowed_roles = {TokenRole.ALIPHATIC_ATOM, TokenRole.AROMATIC_ATOM, TokenRole.BRACKET_ATOM, TokenRole.BOND}
        
        eligible = [i for i in editable if classify_token(tokens[i]) in allowed_roles]
        return eligible

    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        eligible = self._eligible_deletion_positions(tokens)
        if len(eligible) < self.attack_budget:
            return []

        outcomes = []
        seen = set()

        attempts = 0
        max_attempts = n_variants * 10

        while len(outcomes) < n_variants and attempts < max_attempts:
            attempts += 1
            
            # Select positions to delete without replacement
            positions = sorted(self.rng.choice(eligible, size=self.attack_budget, replace=False), reverse=True)
            
            new_tokens = list(tokens)
            deleted_pos = []
            for p in positions:
                new_tokens.pop(p)
                deleted_pos.append(p)
                
            adv_psmiles = "".join(new_tokens)
            if adv_psmiles in seen:
                continue
                
            seen.add(adv_psmiles)
            outcomes.append(AttackOutcome(
                original_tokens=tuple(tokens),
                adversarial_tokens=tuple(new_tokens),
                edit_positions=tuple(deleted_pos),
                attack_type="deletion"
            ))

        return outcomes
