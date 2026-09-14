from abc import ABC, abstractmethod
from typing import Dict, Any, List

class OracleBackend(ABC):
    """
    Abstract base class defining the contract for HPC quantum chemistry oracles.
    This ensures that when an HPC environment is available, the physical evaluation
    can be plugged in without altering the adversarial search loop.
    """
    
    @abstractmethod
    def validate_environment(self) -> bool:
        """
        Verify that the backend executable and environment modules exist.
        """
        pass
        
    @abstractmethod
    def construct_input(self, smiles: str, structure_id: str, config: Dict[str, Any]) -> str:
        """
        Convert a SMILES string to the backend-specific 3D input format
        (e.g., VASP POSCAR/INCAR, QE input block).
        """
        pass
        
    @abstractmethod
    def parse_output(self, output_path: str) -> Dict[str, Any]:
        """
        Extract the target quantity (e.g., bandgap) from the backend's native output.
        """
        pass

class QuantumEspressoBackend(OracleBackend):
    def validate_environment(self) -> bool:
        # Stub
        return False
        
    def construct_input(self, smiles: str, structure_id: str, config: Dict[str, Any]) -> str:
        # Stub
        return ""
        
    def parse_output(self, output_path: str) -> Dict[str, Any]:
        # Stub
        return {}

class VASPBackend(OracleBackend):
    def validate_environment(self) -> bool:
        # Stub
        return False
        
    def construct_input(self, smiles: str, structure_id: str, config: Dict[str, Any]) -> str:
        # Stub
        return ""
        
    def parse_output(self, output_path: str) -> Dict[str, Any]:
        # Stub
        return {}
