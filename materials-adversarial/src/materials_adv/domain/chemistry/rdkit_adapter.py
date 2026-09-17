from typing import Any
try:
    from rdkit import Chem
except ImportError:
    Chem = None
from materials_adv.framework.interfaces import RepresentationAdapter, ValidityChecker, Candidate

class SMILESAdapter(RepresentationAdapter):
    def decode(self, representation: str) -> Candidate:
        if Chem is None:
            raise RuntimeError("RDKit is not installed.")
        # Just return representation for now
        return Candidate(identifier=representation)
        
    def encode(self, candidate: Candidate) -> str:
        if Chem is None:
            raise RuntimeError("RDKit is not installed.")
        return str(candidate.identifier)

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
