"""Probabilistic MCMC Attack Generator (Phase 3)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from rdkit import Chem

from .base import BaseAttack, AttackOutcome
from .registry import register_attack
from ..validation.pipeline import validate

# We need the other attacks for proposals
from .substitution import SubstitutionAttack
from .insertion import InsertionAttack
from .deletion import DeletionAttack

@register_attack("probabilistic_mcmc")
class ProbabilisticMCMCAttack(BaseAttack):
    """MCMC-based attack generator that uses deterministic attacks as proposals.
    
    Proposes local modifications, scores them via chemical heuristics and
    target model prediction drift, and accepts/rejects via Metropolis-Hastings.
    """
    
    def __init__(
        self,
        rng: np.random.Generator,
        predictor: Any = None,
        allowed_tokens: list[str] = None,
        steps: int = 50,
        temperature: float = 10.0,
        protect_attachments: bool = True,
        protect_ring_closures: bool = True,
        protect_branches: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            rng,
            protect_attachments=protect_attachments,
            protect_ring_closures=protect_ring_closures,
            protect_branches=protect_branches,
            **params,
        )
        self.predictor = predictor
        self.allowed_tokens = allowed_tokens or []
        self.steps = steps
        self.temperature = temperature
        
        # Initialize proposal generators
        self._proposals = [
            SubstitutionAttack(rng, allowed_tokens=self.allowed_tokens, n_edits=1, role_preserving=True, protect_attachments=protect_attachments, protect_ring_closures=protect_ring_closures, protect_branches=protect_branches),
            InsertionAttack(rng, allowed_tokens=self.allowed_tokens, n_edits=1, protect_attachments=protect_attachments, protect_ring_closures=protect_ring_closures, protect_branches=protect_branches),
            DeletionAttack(rng, n_edits=1, protect_attachments=protect_attachments, protect_ring_closures=protect_ring_closures, protect_branches=protect_branches)
        ]

    def _score_sequence(self, psmiles: str, original_pred: float) -> tuple[float, float, float]:
        """Returns (total_score, chemical_score, attack_score)"""
        # Chemical heuristic (1.0 if valid RDKit parse, 0.0 otherwise)
        mol = Chem.MolFromSmiles(psmiles)
        chemical_score = 1.0 if mol is not None else -100.0  # severely penalize invalid syntax
        
        # Attack score: prediction drift
        attack_score = 0.0
        if self.predictor is not None and mol is not None:
            pred = float(self.predictor.predict([psmiles])[0])
            attack_score = abs(pred - original_pred)
            
        total_score = chemical_score + attack_score
        return total_score, chemical_score, attack_score

    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        if not self.predictor:
            # Cannot score without a predictor
            return []
            
        original_psmiles = "".join(tokens)
        original_pred = float(self.predictor.predict([original_psmiles])[0])
        
        current_tokens = list(tokens)
        current_psmiles = original_psmiles
        current_score, _, _ = self._score_sequence(current_psmiles, original_pred)
        
        outcomes = []
        seen = {original_psmiles}
        
        # MCMC Loop
        for _ in range(self.steps):
            if len(outcomes) >= n_variants:
                break
                
            # 1. Propose
            proposal_attack = self.rng.choice(self._proposals)
            proposals = proposal_attack.generate(current_tokens, n_variants=1)
            
            if not proposals:
                continue
                
            candidate_outcome = proposals[0]
            candidate_psmiles = candidate_outcome.adversarial_psmiles
            
            # 2. Score
            new_score, chem_score, att_score = self._score_sequence(candidate_psmiles, original_pred)
            
            # 3. Accept/Reject (Metropolis)
            delta = new_score - current_score
            accept_prob = 1.0 if delta > 0 else math.exp(delta / max(self.temperature, 1e-3))
            
            if self.rng.random() < accept_prob:
                # Accept
                current_tokens = list(candidate_outcome.adversarial_tokens)
                current_psmiles = candidate_psmiles
                current_score = new_score
                
                if current_psmiles not in seen and chem_score > 0:
                    seen.add(current_psmiles)
                    
                    # Record the outcome with required MCMC logging fields
                    # The Phase 3 spec requires storing sampling_probability, chemical_constraint_score, and attack_score
                    params = dict(candidate_outcome.params)
                    params.update({
                        "mcmc_accepted": True,
                        "sampling_probability": accept_prob,
                        "chemical_constraint_score": chem_score,
                        "attack_score": att_score,
                    })
                    
                    outcomes.append(
                        AttackOutcome(
                            original_tokens=tuple(tokens),
                            adversarial_tokens=tuple(current_tokens),
                            attack_type=self.name,
                            params=params
                        )
                    )
                    
        return outcomes
