import pytest
import pandas as pd
from materials_adv.validation.sanity_checks import LeakageGuard, TargetSanityGuard, AttackSanityGuard

def test_disjoint_splits_clean():
    splits = {"train": [1, 2, 3], "val": [4, 5], "test": [6, 7]}
    assert LeakageGuard.check_disjoint_splits(splits) == True

def test_disjoint_splits_leakage():
    splits = {"train": [1, 2, 3], "test": [3, 4]}
    with pytest.raises(ValueError):
        LeakageGuard.check_disjoint_splits(splits)

def test_metadata_lineage_clean():
    config = {"property": "Bandgap", "units": "eV"}
    assert TargetSanityGuard.verify_metadata_lineage(config, "Bandgap", "eV") == True

def test_metadata_lineage_mismatch():
    config = {"target_property": "Tg", "units": "K"}
    with pytest.raises(ValueError):
        TargetSanityGuard.verify_metadata_lineage(config, "Bandgap", "eV")

def test_representation_preserving_valid():
    orig = "CC(=O)O"
    adv = "O=C(O)C"
    assert AttackSanityGuard.verify_representation_preserving(orig, adv) == True

def test_representation_preserving_invalid():
    orig = "CC(=O)O"
    adv = "CC(=O)N"
    assert AttackSanityGuard.verify_representation_preserving(orig, adv) == False

def test_chemistry_changed_and_valid():
    orig = "CC(=O)O"
    adv = "CC(=O)N"
    assert AttackSanityGuard.verify_chemistry_changed_and_valid(orig, adv) == True

def test_chemistry_changed_invalid():
    orig = "CC(=O)O"
    adv = "O=C(O)C" # Chemically identical
    assert AttackSanityGuard.verify_chemistry_changed_and_valid(orig, adv) == False

def test_label_inheritance_guard_clean():
    training = [
        {"polymer_id": "p1", "attack_type": "substitution", "property_value": 4.5}
    ]
    orig = {"p1": 5.0} # Target was modified! (e.g. by soft teacher)
    assert AttackSanityGuard.check_no_label_inheritance_in_training_record(training, orig) == True

def test_label_inheritance_guard_violation():
    training = [
        {"polymer_id": "p1", "attack_type": "substitution", "property_value": 5.0}
    ]
    orig = {"p1": 5.0} # Inherited ground truth!
    with pytest.raises(ValueError):
        AttackSanityGuard.check_no_label_inheritance_in_training_record(training, orig)
