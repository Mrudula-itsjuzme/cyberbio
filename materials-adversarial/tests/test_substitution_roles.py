import numpy as np
import pytest
from materials_adv.attacks.substitution import SubstitutionAttack

def test_substitution_roles_bonds():
    rng = np.random.default_rng(42)
    allowed = ["=", "#", ".", "C", "c", "*"]
    
    attack = SubstitutionAttack(
        rng=rng, allowed_tokens=allowed, n_edits=1, role_preserving=True
    )
    
    # Candidates for "=" should only be "#", NOT "."
    candidates = attack._candidates_for("=")
    assert "#" in candidates
    assert "." not in candidates
    
    # Candidates for "." should be empty in this pool (no other disconnects)
    candidates_dot = attack._candidates_for(".")
    assert candidates_dot == ()

def test_substitution_roles_atoms():
    rng = np.random.default_rng(42)
    allowed = ["C", "O", "N", "c", "n", "o", "Cl"]
    
    attack = SubstitutionAttack(
        rng=rng, allowed_tokens=allowed, n_edits=1, role_preserving=True
    )
    
    # Aliphatic to aliphatic only
    candidates_c = attack._candidates_for("C")
    assert "O" in candidates_c
    assert "N" in candidates_c
    assert "Cl" in candidates_c
    assert "c" not in candidates_c
    assert "n" not in candidates_c
    
    # Aromatic to aromatic only
    candidates_aromatic = attack._candidates_for("c")
    assert "n" in candidates_aromatic
    assert "o" in candidates_aromatic
    assert "C" not in candidates_aromatic

def test_protected_tokens_not_substituted():
    rng = np.random.default_rng(42)
    allowed = ["C", "*", "=", "1"]
    
    attack = SubstitutionAttack(
        rng=rng, allowed_tokens=allowed, n_edits=1, role_preserving=True
    )
    
    # Attack on a token list containing protected endpoints
    # It should not even consider '*' or '1'
    tokens = ["*", "C", "C", "1"]
    outcomes = attack.generate(tokens, n_variants=1)
    
    if outcomes:
        for r in outcomes:
            # Edit position should not be 0 or 3
            assert r.edit_positions[0] not in [0, 3]
