import hashlib
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

class ConstructionMode(str, Enum):
    PERIODIC_PREBUILT = "PERIODIC_PREBUILT"
    CAPPED_OLIGOMER_PREBUILT = "CAPPED_OLIGOMER_PREBUILT"
    EXTERNAL_STRUCTURE = "EXTERNAL_STRUCTURE"

def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

class StructurePreparer:
    """
    Contract for preparing physical structures before feeding them to HPC backends.
    Maintains provenance and avoids automatic, unverified polymer wildcard conversions.
    """
    
    @staticmethod
    def prepare_external_structure(
        structure_path: str,
        source_psmiles: str,
        mode: ConstructionMode = ConstructionMode.EXTERNAL_STRUCTURE,
        charge: float = 0.0,
        multiplicity: int = 1,
        cell_metadata: Optional[Dict[str, Any]] = None,
        provenance: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Prepares a structure provided externally. 
        """
        path = Path(structure_path)
        if not path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")
            
        file_hash = get_file_hash(path)
        
        return {
            "structure_path": structure_path,
            "structure_hash": file_hash,
            "construction_mode": mode.value,
            "source_psmiles": source_psmiles,
            "charge": charge,
            "multiplicity": multiplicity,
            "cell_metadata": cell_metadata or {},
            "provenance": provenance
        }
