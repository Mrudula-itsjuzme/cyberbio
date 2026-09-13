import pytest
from rdkit import Chem
import numpy as np

def test_polymer_attachment_parsing():
    """Verify that [*] is parsed as dummy atom (atomic number 0)."""
    smi = "[*]Cc1nc(-c2ccc(-c3csc([*])n3)cc2)cs1"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, "Failed to parse PSMILES"
    
    dummy_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)
    assert dummy_count == 2, f"Expected 2 dummy atoms, got {dummy_count}"
    
    # Check that canonicalization preserves dummy atoms
    can_smi = Chem.MolToSmiles(mol, canonical=True)
    assert "[*]" in can_smi or "*" in can_smi, "Canonicalization destroyed dummy atoms"
    
def test_graph_isomorphism():
    """Verify that canonically equivalent SMILES map to equivalent graph representations."""
    # Let's write a simple node feature extractor
    def get_node_features(mol):
        # Return sorted atomic numbers for quick isomorphism check
        return sorted([a.GetAtomicNum() for a in mol.GetAtoms()])
        
    smi1 = "CC(C)C"
    smi2 = "C(C)(C)C"
    
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    
    assert get_node_features(mol1) == get_node_features(mol2)

def get_full_node_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        int(atom.GetHybridization()),
        atom.GetTotalNumHs(),
        int(atom.GetAtomicNum() == 0)
    ]

def test_graph_feature_dimensions():
    smi = "[*]CC([*])"
    mol = Chem.MolFromSmiles(smi)
    features = [get_full_node_features(a) for a in mol.GetAtoms()]
    assert len(features) == 4
    for f in features:
        assert len(f) == 7
