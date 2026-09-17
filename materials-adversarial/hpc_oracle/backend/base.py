from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class EnvironmentStatus:
    available: bool
    executable_path: Optional[str] = None
    version: Optional[str] = None
    pseudopotential_path: Optional[str] = None
    missing_requirements: List[str] = None

    def __post_init__(self):
        if self.missing_requirements is None:
            self.missing_requirements = []

@dataclass
class ParsedOutput:
    converged: bool
    bandgap_eV: Optional[float]
    total_energy_eV: Optional[float]
    fermi_energy_eV: Optional[float]
    warnings: List[str]
    failure_reason: Optional[str]
    runtime_seconds: Optional[float]

class OracleBackend(ABC):
    """
    Abstract base class defining the contract for HPC quantum chemistry oracles.
    """

    @abstractmethod
    def validate_environment(self) -> EnvironmentStatus:
        """
        Verify that the backend executable and environment modules exist.
        """
        pass

    @abstractmethod
    def prepare_structure(self, external_structure_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare structural metadata before writing inputs.
        """
        pass

    @abstractmethod
    def write_input(self, structure_data: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> None:
        """
        Convert structure data to backend-specific input formats and write to output_dir.
        """
        pass

    @abstractmethod
    def parse_output(self, output_dir: str) -> ParsedOutput:
        """
        Extract the target quantity (e.g., bandgap) from the backend's native output.
        """
        pass

    @abstractmethod
    def validate_result(self, parsed: ParsedOutput) -> bool:
        """
        Ensure the result is scientifically valid and physically plausible.
        """
        pass
