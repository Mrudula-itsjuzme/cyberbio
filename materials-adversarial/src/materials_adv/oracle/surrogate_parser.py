import re
from typing import Dict, Any, Optional
import os

class SurrogateParser:
    """
    Parser for extracting SURROGATE KOHN-SHAM ELECTRONIC GAP from Quantum ESPRESSO outputs.
    """
    
    def parse_qe_output(self, filepath: str) -> Dict[str, Any]:
        result = {
            "status": "FAILED",
            "converged": False,
            "surrogate_gap_eV": None,
            "HOMO_or_VBM_eV": None,
            "LUMO_or_CBM_eV": None,
            "total_energy_eV": None,
            "SCF_iterations": None,
            "runtime": None,
            "warnings": [],
            "failure_reason": None
        }
        
        if not os.path.exists(filepath):
            result["failure_reason"] = "File not found"
            return result
            
        with open(filepath, "r") as f:
            content = f.read()
            
        if "JOB DONE." in content:
            result["status"] = "SUCCESS"
            result["converged"] = True
            
        elif "convergence NOT achieved" in content:
            result["status"] = "FAILED"
            result["converged"] = False
            result["failure_reason"] = "SCF nonconvergence"
            
        elif "CRASH" in content or "Error" in content:
            result["status"] = "FAILED"
            result["failure_reason"] = "Backend crash or error"
            
        else:
            result["status"] = "FAILED"
            result["failure_reason"] = "Malformed output or incomplete job"
            
        # Parse total energy
        # !    total energy              =     -123.45678900 Ry
        energy_match = re.search(r"!\s+total energy\s+=\s+([-+]?\d*\.\d+)\s+Ry", content)
        if energy_match:
            ry_to_ev = 13.6056980659
            result["total_energy_eV"] = float(energy_match.group(1)) * ry_to_ev
            
        # Parse HOMO/LUMO for finite oligomer 
        # highest occupied, lowest unoccupied level (ev):     -5.0000    -2.0000
        gap_match = re.search(r"highest occupied, lowest unoccupied level \(ev\):\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)", content)
        
        if gap_match:
            result["HOMO_or_VBM_eV"] = float(gap_match.group(1))
            result["LUMO_or_CBM_eV"] = float(gap_match.group(2))
            result["surrogate_gap_eV"] = result["LUMO_or_CBM_eV"] - result["HOMO_or_VBM_eV"]
        else:
            # Check for metallic
            # highest occupied level (ev):     -5.0000
            metallic_match = re.search(r"highest occupied level \(ev\):\s+([-+]?\d*\.\d+)", content)
            if metallic_match and "lowest unoccupied" not in content:
                result["HOMO_or_VBM_eV"] = float(metallic_match.group(1))
                result["LUMO_or_CBM_eV"] = float(metallic_match.group(1))
                result["surrogate_gap_eV"] = 0.0
                result["warnings"].append("System is metallic / zero-gap")
                if result["status"] == "SUCCESS":
                    result["status"] = "METALLIC"
            elif result["status"] == "SUCCESS":
                result["status"] = "FAILED"
                result["failure_reason"] = "Missing levels in output"
                
        return result
