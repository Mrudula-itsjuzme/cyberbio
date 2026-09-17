import pytest
import os
import yaml
from pathlib import Path
from rdkit import Chem
from materials_adv.domain.chemistry.surrogate_geometry import SurrogateGeometryBuilder

@pytest.fixture
def config_file(tmp_path):
    config = {
        "mode": "CAPPED_OLIGOMER",
        "oligomer_lengths": [2, 3],
        "capping_rule": "[H]",
        "embedding_method": "ETKDG",
        "embedding_seed": 42,
        "num_conformers": 2,
        "force_field": "MMFF94",
        "max_optimization_steps": 100,
        "charge": 0,
        "multiplicity": 1,
        "periodic_chain_axis": None,
        "cell_length": None,
        "vacuum_dimensions": None
    }
    path = tmp_path / "test_surrogate_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)
    
@pytest.fixture
def periodic_config_file(tmp_path):
    config = {
        "mode": "PERIODIC_CHAIN_SURROGATE",
        "oligomer_lengths": [2],
        "capping_rule": "[H]",
        "embedding_method": "ETKDG",
        "embedding_seed": 42,
        "num_conformers": 1,
        "force_field": "MMFF94",
        "max_optimization_steps": 100,
        "periodic_chain_axis": None, # Missing cell parameters
        "cell_length": None,
        "vacuum_dimensions": None
    }
    path = tmp_path / "test_periodic_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)

def test_capped_oligomer_generation(config_file):
    builder = SurrogateGeometryBuilder(config_file)
    smiles = "[*]CC([*])" # Ethylene repeat
    
    # Generate n=2
    mol, uncertainty = builder.build_surrogate(smiles, oligomer_length=2)
    
    assert mol is not None
    assert mol.GetNumConformers() > 0
    # C-C repeat * 2 = 4 carbons. Capping rule is H.
    # Total atoms will include Hydrogens. 
    # C4H10 (Butane)
    assert sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C') == 4
    assert sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'H') == 10
    
    # Wildcards removed
    assert sum(1 for a in mol.GetAtoms() if a.GetSymbol() == '*') == 0
    
    # Uncertainty
    assert "energy_min" in uncertainty
    assert "oligomer_length" in uncertainty
    assert uncertainty["oligomer_length"] == 2
    assert uncertainty["num_conformers_generated"] <= 2
    
def test_periodic_surrogate_blocks_without_cell_params(periodic_config_file):
    builder = SurrogateGeometryBuilder(periodic_config_file)
    smiles = "[*]CC([*])"
    
    mol, uncertainty = builder.build_surrogate(smiles, oligomer_length=2)
    assert mol is None
    assert "error" in uncertainty
    assert "Periodic mode blocks without cell parameters" in uncertainty["error"]
    
def test_invalid_topology(config_file):
    builder = SurrogateGeometryBuilder(config_file)
    # Ambiguous topology (3 attachment points)
    mol, uncertainty = builder.build_surrogate("[*]C([*])([*])", 2)
    assert mol is None
    assert "Ambiguous" in uncertainty["error"]
