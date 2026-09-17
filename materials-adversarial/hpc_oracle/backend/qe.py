import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

from hpc_oracle.backend.base import OracleBackend, EnvironmentStatus, ParsedOutput

class QuantumEspressoBackend(OracleBackend):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def validate_environment(self) -> EnvironmentStatus:
        missing = []
        pw_path = None
        pseudo_path = os.environ.get("ESPRESSO_PSEUDO", self.config.get("pseudo_dir"))
        
        try:
            res = subprocess.run(["which", "pw.x"], capture_output=True, text=True)
            if res.returncode == 0:
                pw_path = res.stdout.strip()
            else:
                missing.append("pw.x executable not found in PATH")
        except Exception:
            missing.append("pw.x executable not found in PATH")
            
        if not pseudo_path or not Path(pseudo_path).exists():
            missing.append(f"Pseudopotential directory not found: {pseudo_path}")
            
        # Do not invent missing parameters
        if "functional" not in self.config:
            missing.append("Missing required parameter: functional")
        if "cutoff_wfc" not in self.config:
            missing.append("Missing required parameter: cutoff_wfc")
            
        return EnvironmentStatus(
            available=len(missing) == 0,
            executable_path=pw_path,
            pseudopotential_path=pseudo_path,
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
        # Skeleton implementation. We do not invent missing physics!
        input_content = f"""&CONTROL
  calculation = 'scf'
  pseudo_dir  = '{self.config.get("pseudo_dir", "./")}'
/
&SYSTEM
  ecutwfc = {self.config.get("cutoff_wfc", "MISSING_CUTOFF")}
  ! Functional: {self.config.get("functional", "MISSING_FUNCTIONAL")}
/
&ELECTRONS
  conv_thr = {self.config.get("conv_thr", 1e-6)}
/
! STRUCTURE INSERTED FROM: {structure_data.get('source_path')}
"""
        with open(out_path / "pw.scf.in", "w") as f:
            f.write(input_content)

    def parse_output(self, output_dir: str) -> ParsedOutput:
        out_path = Path(output_dir) / "pw.scf.out"
        if not out_path.exists():
            return ParsedOutput(False, None, None, None, [], "Output file not found", None)
            
        converged = False
        warnings = []
        failure_reason = None
        
        with open(out_path) as f:
            content = f.read()
            if "convergence NOT achieved" in content:
                failure_reason = "SCF convergence failed"
            elif "JOB DONE" in content:
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
