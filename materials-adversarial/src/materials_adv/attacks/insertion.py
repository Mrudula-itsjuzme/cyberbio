from typing import Sequence

import numpy as np

from materials_adv.attacks.base import BaseAttack
from materials_adv.attacks.generator import AttackOutcome
from materials_adv.attacks.registry import register_attack
from materials_adv.attacks.token_space import TokenRole, classify_token, editable_positions


@register_attack("insertion")
class InsertionAttack(BaseAttack):
    def __init__(
        self,
        rng: np.random.Generator,
        allowed_tokens: Sequence[str],
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
        
        # Allowed insertion pool: only atoms and bonds
        allowed_roles = {TokenRole.ALIPHATIC_ATOM, TokenRole.AROMATIC_ATOM, TokenRole.BRACKET_ATOM, TokenRole.BOND}
        self.allowed_pool = [t for t in allowed_tokens if classify_token(t) in allowed_roles]

    def _eligible_insertion_positions(self, tokens: list[str]) -> list[int]:
        """Positions where an insertion can occur.
        
        We allow insertion at any index `i` (meaning before `tokens[i]`) if either `i`
        or `i-1` is an editable position. We also allow insertion at the very end `len(tokens)`
        if the last token is editable.
        """
        editable = set(editable_positions(
            tokens,
            protect_attachments=self.protect_attachments,
            protect_ring_closures=self.protect_ring_closures,
            protect_branches=self.protect_branches
        ))
        
        eligible = []
        for i in range(len(tokens) + 1):
            if (i in editable) or (i - 1 in editable):
                eligible.append(i)
        return eligible

    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        if not self.allowed_pool:
            return []
            
        eligible = self._eligible_insertion_positions(tokens)
        if not eligible:
            return []

        outcomes = []
        seen = set()

        # Try generating candidates
        attempts = 0
        max_attempts = n_variants * 10

        while len(outcomes) < n_variants and attempts < max_attempts:
            attempts += 1
            
            # Pick insertion positions (can pick the same position multiple times if attack_budget > 1)
            # We sort them descending so that inserting doesn't mess up earlier indices.
            positions = sorted(self.rng.choice(eligible, size=self.attack_budget, replace=True), reverse=True)
            new_tokens = list(tokens)
            
            inserted_pos = []
            for p in positions:
                token_to_insert = self.rng.choice(self.allowed_pool)
                new_tokens.insert(p, token_to_insert)
                inserted_pos.append(p)
                
            adv_psmiles = "".join(new_tokens)
            if adv_psmiles in seen:
                continue
                
            seen.add(adv_psmiles)
            outcomes.append(AttackOutcome(
                original_tokens=tuple(tokens),
                adversarial_tokens=tuple(new_tokens),
                edit_positions=tuple(inserted_pos),
                attack_type="insertion"
            ))

        return outcomes
