import pytest
from materials_adv.domain.chemistry.polymer_repeat import PolymerRepeatGraph
from rdkit import Chem

def test_two_attachment_point_detection():
    # Valid
    p = PolymerRepeatGraph("[*]CC([*])")
    assert p.is_unambiguous
    assert len(p.attachment_indices) == 2
    
    # Invalid (3)
    p3 = PolymerRepeatGraph("[*]C([*])C([*])")
    assert not p3.is_unambiguous
    assert len(p3.attachment_indices) == 3

def test_repeat_connectivity_generation():
    p = PolymerRepeatGraph("[*]CC([*])")
    o2 = p.build_oligomer(2)
    assert o2 is not None
    # C-C repeat unit. 2 carbons * 2 = 4 carbons. Attachments are removed.
    assert o2.GetNumAtoms() == 4
    
def test_no_3d_geometry_in_topology_mode():
    p = PolymerRepeatGraph("[*]c1ccc([*])cc1")
    res = p.analyze()
    assert not res["unique_3d_periodic_recoverable"]
    assert not res["unique_periodic_cell_recoverable"]
    assert res["unique_2d_graph_recoverable"]
    
def test_valence_preservation():
    p = PolymerRepeatGraph("[*]O[*]") # This would form a peroxide chain -O-O-O-
    o3 = p.build_oligomer(3)
    assert o3 is not None
    # 1 oxygen per unit, 3 units -> 3 oxygens. 
    assert o3.GetNumAtoms() == 3
    # Check bonds
    for bond in o3.GetBonds():
        assert bond.GetBondType() == Chem.BondType.SINGLE

def test_deterministic_hash():
    p1 = PolymerRepeatGraph("[*]CC([*])")
    p2 = PolymerRepeatGraph("[*]CC([*])")
    assert p1.get_topology_hash() == p2.get_topology_hash()
    assert p1.get_topology_hash() != "INVALID"
    assert p1.get_topology_hash() != "ERROR"
