"""True label-preserving control via SMILES randomization."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from materials_adv.attacks.base import BaseAttack, AttackOutcome
from materials_adv.attacks.registry import register_attack

@register_attack("randomization")
class SmilesRandomizationAttack(BaseAttack):
    """Generates structurally identical but syntactically different strings.
    
    This provides the only true chemically-invariant, label-preserving control.
    """
    
    def generate(self, tokens: list[str], n_variants: int = 1) -> list[AttackOutcome]:
        from materials_adv.data.tokenizer import tokenize
        
        original_representation = "".join(tokens)
        mol = Chem.MolFromSmiles(original_representation)
        if mol is None:
            return []
            
        outcomes = []
        seen = {original_representation}

        # Use RDKit's explicitly seeded vector API. MolToSmiles(doRandom=True)
        # uses global state and made an identically seeded experiment produce a
        # different primary control candidate bank across processes.
        rdkit_seed = int(self.rng.integers(1, 2**31 - 1))
        randomized = Chem.MolToRandomSmilesVect(
            mol, n_variants * 10, randomSeed=rdkit_seed
        )
        for rand_smiles in randomized:
            if len(outcomes) >= n_variants:
                break
            if rand_smiles not in seen:
                seen.add(rand_smiles)
                new_tokens = tuple(tokenize(rand_smiles))
                outcomes.append(
                    AttackOutcome(
                        original_tokens=tuple(tokens),
                        adversarial_tokens=new_tokens,
                        attack_type=self.name,
                        params={"rdkit_random_seed": rdkit_seed},
                    )
                )
                
        return outcomes
