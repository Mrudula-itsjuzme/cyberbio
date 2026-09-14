from typing import Any
try:
    from rdkit import Chem
except ImportError:
    Chem = None
from materials_adv.framework.interfaces import RepresentationAdapter, ValidityChecker

class SMILESAdapter(RepresentationAdapter):
    def to_object(self, representation: str) -> Any:
        if Chem is None:
            raise RuntimeError("RDKit is not installed.")
        return Chem.MolFromSmiles(representation)
        
    def to_representation(self, obj: Any) -> str:
        if Chem is None:
            raise RuntimeError("RDKit is not installed.")
        return Chem.MolToSmiles(obj)

class RDKitValidityChecker(ValidityChecker):
    def is_valid(self, obj: Any) -> bool:
        """
        Check if the RDKit Mol object is valid.
        """
        if obj is None:
            return False
        try:
            Chem.SanitizeMol(obj)
            return True
        except Exception:
            return False
