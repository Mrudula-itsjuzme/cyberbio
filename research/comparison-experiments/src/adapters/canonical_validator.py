import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent / "materials-adversarial"
if str(repo_root) not in sys.path and repo_root.exists():
    sys.path.insert(0, str(repo_root / "src"))

from materials_adv.domain.chemistry.plausibility import ChemicalPlausibilityValidator
from materials_adv.domain.chemistry.rdkit_adapter import RDKitValidityChecker

class CanonicalValidatorAdapter:
    def __init__(self, check_plausibility: bool = True):
        self.rdkit_checker = RDKitValidityChecker()
        if check_plausibility:
            self.plausibility_checker = ChemicalPlausibilityValidator()
        else:
            self.plausibility_checker = None

    def is_valid_rdkit(self, sequence: str) -> bool:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(sequence)
        if mol is None:
            return False
        return self.rdkit_checker.is_valid(mol)

    def is_valid_plausible(self, sequence: str) -> bool:
        if not self.is_valid_rdkit(sequence):
            return False
        if self.plausibility_checker:
            try:
                # Based on the audit, validate method returns ValidationResult
                result = self.plausibility_checker.validate(sequence)
                return bool(result)
            except Exception:
                return False
        return True
