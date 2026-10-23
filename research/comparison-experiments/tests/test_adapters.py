import pytest
from pathlib import Path
from src.adapters.canonical_validator import CanonicalValidatorAdapter

def test_validator_adapter():
    validator = CanonicalValidatorAdapter(check_plausibility=True)
    
    # Valid
    valid_seq = "[*]CC([*])c1ccccc1"
    assert validator.is_valid_rdkit(valid_seq)
    
    # Invalid syntax
    invalid_seq = "C(C"
    assert not validator.is_valid_rdkit(invalid_seq)
