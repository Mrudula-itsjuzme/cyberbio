from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class OracleUncertainty:
    numerical_convergence_uncertainty: Optional[float]
    method_mismatch_uncertainty: Optional[float]
    construction_uncertainty: Optional[float]
    calibration_residual_scale: Optional[float]
    
    def to_dict(self):
        return {
            "numerical_convergence_uncertainty": self.numerical_convergence_uncertainty if self.numerical_convergence_uncertainty is not None else "UNKNOWN",
            "method_mismatch_uncertainty": self.method_mismatch_uncertainty if self.method_mismatch_uncertainty is not None else "UNKNOWN",
            "construction_uncertainty": self.construction_uncertainty if self.construction_uncertainty is not None else "UNKNOWN",
            "calibration_residual_scale": self.calibration_residual_scale if self.calibration_residual_scale is not None else "UNKNOWN",
        }
        
    @classmethod
    def from_dict(cls, data):
        def parse_val(v):
            return v if v != "UNKNOWN" else None
        return cls(
            numerical_convergence_uncertainty=parse_val(data.get("numerical_convergence_uncertainty")),
            method_mismatch_uncertainty=parse_val(data.get("method_mismatch_uncertainty")),
            construction_uncertainty=parse_val(data.get("construction_uncertainty")),
            calibration_residual_scale=parse_val(data.get("calibration_residual_scale")),
        )
