import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from hpc_oracle.backend.base import OracleBackend, EnvironmentStatus, ParsedOutput

class VASPBackend(OracleBackend):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def validate_environment(self) -> EnvironmentStatus:
        missing = []
        vasp_path = None
        potcar_dir = os.environ.get("VASP_POTCAR_DIR", self.config.get("potcar_dir"))
        
        try:
            res = subprocess.run(["which", "vasp_std"], capture_output=True, text=True)
            if res.returncode == 0:
                vasp_path = res.stdout.strip()
            else:
                missing.append("vasp_std executable not found in PATH")
        except Exception:
            missing.append("vasp_std executable not found in PATH")
            
        if not potcar_dir or not Path(potcar_dir).exists():
            missing.append(f"POTCAR directory not found: {potcar_dir}")
            
        if "functional" not in self.config:
            missing.append("Missing required parameter: functional")
        if "encut" not in self.config:
            missing.append("Missing required parameter: encut")
            
        return EnvironmentStatus(
            available=len(missing) == 0,
            executable_path=vasp_path,
            pseudopotential_path=potcar_dir,
            missing_requirements=missing
        )

    def prepare_structure(self, external_structure_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(external_structure_path)
        if not path.exists():
            raise FileNotFoundError(f"Structure file not found: {external_structure_path}")
        return {
            "source_path": external_structure_path,
            "format": path.suffix.strip(".")
        }

    def write_input(self, structure_data: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Skeleton INCAR
        incar = f"""System = Materials Adversarial
ENCUT = {self.config.get("encut", "MISSING_ENCUT")}
GGA = {self.config.get("functional", "MISSING_FUNCTIONAL")}
EDIFF = {self.config.get("ediff", 1e-6)}
"""
        with open(out_path / "INCAR", "w") as f:
            f.write(incar)
            
        with open(out_path / "KPOINTS", "w") as f:
            f.write("Automatic mesh\\n0\\nGamma\\n1 1 1\\n0 0 0\\n")
            
        # Write dummy POSCAR for structural prep if missing
        with open(out_path / "POSCAR", "w") as f:
            f.write(f"Structure from {structure_data.get('source_path')}\\n")
            
        # Ensure we DO NOT generate fake POTCARs.
        pass

    def parse_output(self, output_dir: str) -> ParsedOutput:
        out_path = Path(output_dir) / "OUTCAR"
        if not out_path.exists():
            return ParsedOutput(False, None, None, None, [], "OUTCAR file not found", None)
            
        converged = False
        warnings = []
        failure_reason = None
        
        with open(out_path) as f:
            content = f.read()
            if "General failing" in content:
                failure_reason = "SCF failing"
            elif "Voluntary context switches" in content:
                converged = True
                
        return ParsedOutput(
            converged=converged,
            bandgap_eV=None, # Skeleton parser
            total_energy_eV=None,
            fermi_energy_eV=None,
            warnings=warnings,
            failure_reason=failure_reason,
            runtime_seconds=None
        )

    def validate_result(self, parsed: ParsedOutput) -> bool:
        if not parsed.converged or parsed.failure_reason is not None:
            return False
        return True
