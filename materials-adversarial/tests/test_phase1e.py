import numpy as np
import pytest

from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.token_space import TokenRole, classify_token

def test_insertion_attack():
    rng = np.random.default_rng(42)
    # Give a small pool
    allowed = ["C", "O", "="]
    attack = InsertionAttack(rng, allowed, n_edits=1)
    
    tokens = ["C", "C"]
    outcomes = attack.generate(tokens, n_variants=2)
    
    for r in outcomes:
        assert len(r.adversarial_psmiles) > len("".join(tokens))
        assert r.attack_type == "insertion"
        # ensure no protected token like '.' or '(' was inserted
        inserted = set(r.adversarial_psmiles) - set("".join(tokens))
        for c in inserted:
            assert c in allowed

def test_deletion_attack_protected():
    rng = np.random.default_rng(42)
    attack = DeletionAttack(rng, n_edits=1)
    
    tokens = ["[*]", "C", ".", "(", "1", "C", ")"]
    outcomes = attack.generate(tokens, n_variants=5)
    
    for r in outcomes:
        # Protected elements should NOT be deleted
        assert "[*]" in r.adversarial_psmiles
        assert "." in r.adversarial_psmiles
        assert "(" in r.adversarial_psmiles
        assert "1" in r.adversarial_psmiles
        assert ")" in r.adversarial_psmiles

def test_rearrangement_attack():
    rng = np.random.default_rng(42)
    attack = RearrangementAttack(rng, window_size=3)
    
    tokens = ["[*]", "C", "=", "O", "C", "[*]"]
    outcomes = attack.generate(tokens, n_variants=1)
    
    for r in outcomes:
        assert "[*]" in r.adversarial_psmiles
        # Length remains identical
        assert len("".join(tokens)) == len(r.adversarial_psmiles)
        # Contents identical just permuted
        from collections import Counter
        assert Counter(r.adversarial_psmiles) == Counter("".join(tokens))
        assert "".join(tokens) != r.adversarial_psmiles
