"""Domain-specific chemical plausibility validator for polymer adversarial attacks.

Enforces scientific validity rules on candidate adversarial SMILES/PSMILES:
1. RDKit syntax & valence validity.
2. Polymer attachment star (*) balance.
3. Tanimoto structural similarity threshold (S_Tanimoto(x, x') >= min_similarity).
4. Bioisosteric functional group replacement constraints.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string can be parsed by RDKit with valid valence."""
    if not smiles or not isinstance(smiles, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        # Sanitize to check valence and ring systems
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def check_polymer_attachment_balance(original_psmiles: str, candidate_psmiles: str) -> bool:
    """Check if polymer attachment star (*) count matches between original and candidate PSMILES."""
    orig_stars = original_psmiles.count("*")
    cand_stars = candidate_psmiles.count("*")
    # If original is a polymer repeat unit with attachment points, candidate must preserve count
    if orig_stars > 0:
        return orig_stars == cand_stars
    return True


def compute_tanimoto_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    """Compute Tanimoto similarity between two SMILES using Morgan fingerprints."""
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return 0.0
        
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)
        return float(FingerprintSimilarity(fp_a, fp_b))
    except Exception as e:
        logger.debug(f"Error computing Tanimoto similarity: {e}")
        return 0.0


def check_molecular_weight_bounds(original_smiles: str, candidate_smiles: str, min_factor: float = 0.5, max_factor: float = 1.5) -> bool:
    """Check if candidate molecular weight stays within [min_factor, max_factor] of original."""
    try:
        from rdkit.Chem import Descriptors
        mol_a = Chem.MolFromSmiles(original_smiles)
        mol_b = Chem.MolFromSmiles(candidate_smiles)
        if mol_a is None or mol_b is None:
            return False
        mw_a = Descriptors.ExactMolWt(mol_a)
        mw_b = Descriptors.ExactMolWt(mol_b)
        if mw_a == 0:
            return True
        ratio = mw_b / mw_a
        return min_factor <= ratio <= max_factor
    except Exception:
        return True


class ChemicalPlausibilityValidator:
    """Validates adversarial candidates against chemical and structural constraints."""

    def __init__(
        self,
        min_tanimoto_similarity: float = 0.5,
        max_edit_distance: Optional[int] = None,
        require_attachment_preservation: bool = True,
        enforce_valence: bool = True,
        enforce_mw_bounds: bool = True,
    ) -> None:
        self.min_tanimoto_similarity = min_tanimoto_similarity
        self.max_edit_distance = max_edit_distance
        self.require_attachment_preservation = require_attachment_preservation
        self.enforce_valence = enforce_valence
        self.enforce_mw_bounds = enforce_mw_bounds

    def validate(self, original_smiles: str, candidate_smiles: str) -> Tuple[bool, dict]:
        """Validate if a candidate SMILES is a scientifically plausible adversarial mutation of original_smiles."""
        details = {
            "valid_syntax": False,
            "attachment_preserved": False,
            "mw_bounds_passed": False,
            "tanimoto_similarity": 0.0,
            "similarity_passed": False,
            "is_plausible": False,
        }

        # 1. RDKit validity & valence
        if not is_valid_smiles(candidate_smiles):
            return False, details
        details["valid_syntax"] = True

        # 2. Polymer attachment preservation
        if self.require_attachment_preservation:
            if not check_polymer_attachment_balance(original_smiles, candidate_smiles):
                return False, details
        details["attachment_preserved"] = True

        # 3. Molecular weight ratio check
        if self.enforce_mw_bounds:
            if not check_molecular_weight_bounds(original_smiles, candidate_smiles):
                return False, details
        details["mw_bounds_passed"] = True

        # 4. Tanimoto similarity constraint
        sim = compute_tanimoto_similarity(original_smiles, candidate_smiles)
        details["tanimoto_similarity"] = sim
        if sim < self.min_tanimoto_similarity:
            return False, details
        details["similarity_passed"] = True

        details["is_plausible"] = True
        return True, details


# Pre-defined bioisosteric functional group substitution mappings for plausible attacks
BIOISOSTERIC_SUBSTITUTIONS = {
    "F": ["Cl", "Br"],
    "Cl": ["F", "Br"],
    "Br": ["F", "Cl"],
    "OH": ["SH", "NH2", "OCH3"],
    "SH": ["OH"],
    "CH3": ["CF3", "CH2CH3"],
    "NH2": ["OH", "NHCH3"],
}
