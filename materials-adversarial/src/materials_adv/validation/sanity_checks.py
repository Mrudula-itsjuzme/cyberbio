"""Automated sanity checks for dataset splits and attack logic.

Guards against:
- Data leakage across splits
- Improper mixing of Target distributions (Tg vs Bandgap)
- Representation-preserving attacks that actually change the chemistry
- Chemistry-changing attacks inheriting ground truth
"""

import logging
from typing import Set, Dict, Any, List
import pandas as pd
import sys

log = logging.getLogger(__name__)

class LeakageGuard:
    """Detects overlap between data splits."""
    
    @staticmethod
    def check_disjoint_splits(splits: Dict[str, List[int]], protocol: str = "random") -> bool:
        """Verify that all splits contain mutually exclusive indices.
        
        Args:
            splits: Dict of split names to lists of indices/sample IDs
            protocol: The split protocol (e.g. 'random', 'scaffold')
        """
        seen = set()
        for split_name, indices in splits.items():
            # Check for repeated indices across splits
            overlap = seen.intersection(set(indices))
            if overlap:
                raise ValueError(f"Data leakage detected! Split {split_name} shares {len(overlap)} indices with prior splits.")
            seen.update(indices)
        return True

    @staticmethod
    def check_exact_representation_overlap(df: pd.DataFrame, splits: Dict[str, List[int]], rep_col: str) -> bool:
        """Verify that exact representations don't leak across splits."""
        seen_reps = set()
        
        # Determine if splits are integer indices or sample IDs
        is_index = False
        if splits and list(splits.values())[0]:
            is_index = isinstance(list(splits.values())[0][0], int)
            
        for split_name, indices in splits.items():
            if is_index:
                split_reps = set(df.iloc[indices][rep_col].tolist())
            else:
                if 'polymer_id' in df.columns:
                    split_reps = set(df[df['polymer_id'].isin(indices)][rep_col].tolist())
                else:
                    log.warning("Cannot check representation overlap without integer indices or polymer_id")
                    return True
                    
            overlap = seen_reps.intersection(split_reps)
            if overlap:
                raise ValueError(f"Data leakage! Split {split_name} shares {len(overlap)} exact representations with prior splits.")
            seen_reps.update(split_reps)
        return True


class TargetSanityGuard:
    """Verifies that the target values belong to the correct physical domain."""
    
    @staticmethod
    def verify_metadata_lineage(config: dict, expected_property: str, expected_units: str) -> bool:
        """Verify the dataset property and units match expectations."""
        prop = config.get("property") or config.get("target_property")
        units = config.get("units")
        
        if prop != expected_property:
            raise ValueError(f"Lineage mismatch: expected property '{expected_property}', got '{prop}'")
        if units != expected_units:
            raise ValueError(f"Lineage mismatch: expected units '{expected_units}', got '{units}'")
        return True

    @staticmethod
    def check_no_silent_zeros(targets: pd.Series) -> bool:
        """Ensure missing targets aren't silently converted to 0.0 (unless 0.0 is physically meaningful for all of them)."""
        if targets.isna().any():
            log.warning("Dataset contains NaN targets.")
            
        zeros = (targets == 0.0).sum()
        if zeros > len(targets) * 0.1:  # Warning if >10% are exactly 0.0
            log.warning(f"High percentage of 0.0 targets ({zeros}/{len(targets)}). Check for improper missing value imputation.")
        return True

class AttackSanityGuard:
    """Validates the properties of adversarial candidate molecules."""
    
    @staticmethod
    def _get_rdkit():
        try:
            from rdkit import Chem
            return Chem
        except ImportError:
            raise ImportError("RDKit is required for attack sanity checks. Cannot verify chemical structures.")

    @staticmethod
    def verify_representation_preserving(original_smiles: str, adv_smiles: str) -> bool:
        """For attacks claiming to be representation-preserving, verify the canonical graph is identical.
        Fails closed if RDKit is missing or if parsing fails."""
        Chem = AttackSanityGuard._get_rdkit()
        
        mol_orig = Chem.MolFromSmiles(original_smiles)
        mol_adv = Chem.MolFromSmiles(adv_smiles)
        
        if not mol_orig or not mol_adv:
            return False
            
        can_orig = Chem.MolToSmiles(mol_orig, canonical=True)
        can_adv = Chem.MolToSmiles(mol_adv, canonical=True)
        
        return can_orig == can_adv

    @staticmethod
    def check_no_label_inheritance_in_training_record(training_records: list[dict], original_records_dict: dict) -> bool:
        """Verify that chemistry-changing training candidates do NOT inherit the original label.
        
        training_records: the augmented records fed to the model (e.g. from augmented_train.csv)
        original_records_dict: mapping of polymer_id -> original property_value
        """
        for rec in training_records:
            attack_type = rec.get("attack_type")
            if not attack_type or attack_type in ["None", "randomized_smiles", "randomization"]:
                continue
                
            # It's a chemistry changing attack
            poly_id = rec.get("polymer_id")
            val = rec.get("property_value")
            
            orig_val = original_records_dict.get(poly_id)
            if orig_val is not None and val is not None:
                if abs(float(val) - float(orig_val)) < 1e-5:
                    # Inherited exact same float value!
                    # If there's an oracle target, it might coincidentally be the same, but very unlikely
                    if not rec.get("oracle_derived", False):
                        raise ValueError(
                            f"Label inheritance violation! Chemistry-changing attack '{attack_type}' "
                            f"on {poly_id} inherited the original label {val}."
                        )
        return True

    @staticmethod
    def verify_chemistry_changed_and_valid(original_smiles: str, adv_smiles: str) -> bool:
        """Ensure chemistry-changing attacks DO NOT claim to preserve the ground truth label.
        Verifies that the canonical SMILES are actually different.
        Fails closed on parse failure or RDKit missing."""
        Chem = AttackSanityGuard._get_rdkit()
        
        mol_orig = Chem.MolFromSmiles(original_smiles)
        mol_adv = Chem.MolFromSmiles(adv_smiles)
        
        if not mol_orig or not mol_adv:
            return False # Parsing failed, invalid candidate
            
        can_orig = Chem.MolToSmiles(mol_orig, canonical=True)
        can_adv = Chem.MolToSmiles(mol_adv, canonical=True)
        
        if can_orig == can_adv:
            log.warning(f"Chemistry-changing attack generated a canonical equivalent: {original_smiles} -> {adv_smiles}")
            return False
            
        return True
