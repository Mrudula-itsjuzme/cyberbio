from typing import Sequence

import numpy as np

from materials_adv.attacks.base import BaseAttack
from materials_adv.attacks.generator import AttackOutcome
from materials_adv.attacks.registry import register_attack
from materials_adv.attacks.token_space import TokenRole, classify_token, editable_positions


@register_attack("rearrangement")
class RearrangementAttack(BaseAttack):
    def __init__(
        self,
        rng: np.random.Generator,
        window_size: int = 3,
        protect_attachments: bool = True,
        protect_ring_closures: bool = True,
        protect_branches: bool = True,
    ):
        super().__init__(rng)
        self.window_size = window_size
        self.protect_attachments = protect_attachments
        self.protect_ring_closures = protect_ring_closures
        self.protect_branches = protect_branches

    def _eligible_windows(self, tokens: list[str]) -> list[int]:
        """Find starting indices of windows of size k that contain NO protected elements."""
        if len(tokens) < self.window_size:
            return []
            
        editable = set(editable_positions(
            tokens,
            protect_attachments=self.protect_attachments,
            protect_ring_closures=self.protect_ring_closures,
            protect_branches=self.protect_branches
        ))
        
        eligible_starts = []
        for i in range(len(tokens) - self.window_size + 1):
            window_indices = set(range(i, i + self.window_size))
            if window_indices.issubset(editable):
                eligible_starts.append(i)
                
        return eligible_starts

    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        eligible = self._eligible_windows(tokens)
        if not eligible:
            return []

        outcomes = []
        seen = set()

        attempts = 0
        max_attempts = n_variants * 10

        while len(outcomes) < n_variants and attempts < max_attempts:
            attempts += 1
            
            start_idx = self.rng.choice(eligible)
            
            new_tokens = list(tokens)
            window = new_tokens[start_idx : start_idx + self.window_size]
            
            # Randomly permute the window
            permuted = list(self.rng.permutation(window))
            
            if permuted == window:
                continue
                
            new_tokens[start_idx : start_idx + self.window_size] = permuted
            
            adv_psmiles = "".join(new_tokens)
            if adv_psmiles in seen:
                continue
                
            seen.add(adv_psmiles)
            outcomes.append(AttackOutcome(
                original_tokens=tuple(tokens),
                adversarial_tokens=tuple(new_tokens),
                edit_positions=(start_idx,),
                attack_type="rearrangement"
            ))

        return outcomes
