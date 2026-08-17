"""True label-preserving control via SMILES randomization."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from .base import BaseAttack, AttackOutcome
from .registry import register_attack

@register_attack("randomization")
class SmilesRandomizationAttack(BaseAttack):
    """Generates structurally identical but syntactically different strings.
    
    This provides the only true chemically-invariant, label-preserving control.
    """
    
    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        from ..data.tokenizer import tokenize
        
        original_psmiles = "".join(tokens)
        mol = Chem.MolFromSmiles(original_psmiles)
        if mol is None:
            return []
            
        outcomes = []
        seen = {original_psmiles}
        
        # RDKit's RNG for doRandom=True is global. We don't have direct control 
        # via the numpy Generator, but we can seed python's/rdkit's if needed, 
        # or just rely on the attempts loop to find unique strings.
        
        for _ in range(n_variants * 10):
            if len(outcomes) >= n_variants:
                break
                
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            if rand_smiles not in seen:
                seen.add(rand_smiles)
                new_tokens = tuple(tokenize(rand_smiles))
                outcomes.append(
                    AttackOutcome(
                        original_tokens=tuple(tokens),
                        adversarial_tokens=new_tokens,
                        attack_type=self.name,
                    )
                )
                
        return outcomes
