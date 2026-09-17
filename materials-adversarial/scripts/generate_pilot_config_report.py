import yaml
import json
import os

def generate_report():
    config_path = "hpc_oracle/configs/calibration_pilot_qe.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    report = []
    
    # Classifications
    classifications = {
        "backend": {"status": "KNOWN", "reason": "Explicitly defined implementation (QE)"},
        "oracle_class": {"status": "KNOWN", "reason": "Explicitly defined mapping subset"},
        "target_quantity": {"status": "KNOWN", "reason": "Matched to Phase 13 Bandgap definition"},
        "functional": {"status": "DATASET_UNKNOWN", "reason": "Original polyVERSE DFT functional is missing in metadata."},
        "pseudopotential_family": {"status": "DATASET_UNKNOWN", "reason": "Original polyVERSE pseudopotential is missing in metadata."},
        "cutoff_wfc": {"status": "DATASET_UNKNOWN", "reason": "Cutoff missing; dependent on pseudo."},
        "cutoff_rho": {"status": "DATASET_UNKNOWN", "reason": "Cutoff missing; dependent on pseudo."},
        "k_points": {"status": "DATASET_UNKNOWN", "reason": "BZ sampling density for 1D systems is not detailed in polyVERSE source."},
        "smearing": {"status": "DATASET_UNKNOWN", "reason": "Smearing protocol missing."},
        "geometry_optimization": {"status": "DATASET_UNKNOWN", "reason": "Relaxation protocol unknown."},
        "periodicity": {"status": "KNOWN", "reason": "1D chain constraints known from source metadata."},
        "charge": {"status": "KNOWN", "reason": "Defaulting to neutral systems."},
        "multiplicity": {"status": "KNOWN", "reason": "Defaulting to singlet ground states."},
        "convergence_threshold": {"status": "DATASET_UNKNOWN", "reason": "Thresholds not published."},
        "structure_mode": {"status": "KNOWN", "reason": "Defines building pipeline constraint."},
        "scheduler_resources": {"status": "CLUSTER_REQUIRED", "reason": "Requires matching to specific HPC environment limits."}
    }
    
    for key, cinfo in classifications.items():
        val = config.get(key)
        if val == "UNKNOWN":
            val = None
        report.append({
            "parameter": key,
            "status": cinfo["status"],
            "configured_value": val,
            "explanation": cinfo["reason"]
        })
        
    out_dir = "results/oracle_calibration_pilot"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "config_completeness.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Generated config completeness report at {out_path}")

if __name__ == "__main__":
    generate_report()
