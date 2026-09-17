import pytest
import os
import json
from pathlib import Path
from hpc_oracle.backend import QuantumEspressoBackend, VASPBackend, EnvironmentStatus, ParsedOutput
from hpc_oracle.structure.prep import StructurePreparer, ConstructionMode
from hpc_oracle.slurm.generator import generate_slurm_script
from hpc_oracle.manifests.calibration import generate_calibration_manifest
from hpc_oracle.analysis.uncertainty import OracleUncertainty
import pandas as pd

def test_qe_validation_missing():
    # Should flag missing functional/cutoff
    backend = QuantumEspressoBackend({})
    status = backend.validate_environment()
    assert not status.available
    assert "Missing required parameter: functional" in status.missing_requirements

def test_vasp_validation_missing():
    backend = VASPBackend({})
    status = backend.validate_environment()
    assert not status.available
    assert "Missing required parameter: functional" in status.missing_requirements

def test_structure_prep_external(tmp_path):
    # Dummy structure
    struct_path = tmp_path / "dummy.xyz"
    struct_path.write_text("dummy")
    
    res = StructurePreparer.prepare_external_structure(
        str(struct_path), "CCO"
    )
    assert res["construction_mode"] == "EXTERNAL_STRUCTURE"
    assert res["source_psmiles"] == "CCO"
    assert "structure_hash" in res

def test_slurm_generator():
    script = generate_slurm_script(
        job_name="test_job",
        output_dir="/path/to/out",
        executable_path="pw.x",
        input_file="in.scf",
        output_file="out.scf",
        nodes=2,
        array_size=10
    )
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --array=1-10" in script
    assert "srun  pw.x < in.scf > out.scf" in script

def test_calibration_manifest():
    df = pd.DataFrame({"original_representation": ["CCO", "CCC"], "property_value": [1.0, 2.0]})
    jobs = generate_calibration_manifest(df, "hash123", "qe")
    assert len(jobs) == 2
    assert jobs[0]["source_id"] == "CCO"
    assert jobs[0]["backend"] == "qe"
    assert jobs[0]["config_hash"] == "hash123"

def test_uncertainty_fields():
    u = OracleUncertainty(None, 0.1, None, 0.05)
    d = u.to_dict()
    assert d["numerical_convergence_uncertainty"] == "UNKNOWN"
    assert d["method_mismatch_uncertainty"] == 0.1
    
    u2 = OracleUncertainty.from_dict(d)
    assert u2.numerical_convergence_uncertainty is None
    assert u2.method_mismatch_uncertainty == 0.1

def test_pilot_selection_deterministic():
    import subprocess
    # Run pilot selection
    res = subprocess.run([".venv/bin/python", "scripts/select_calibration_pilot.py"], capture_output=True, text=True)
    assert res.returncode == 0
    df = pd.read_csv("hpc_oracle/manifests/calibration_pilot.csv")
    assert len(df) <= 10
    assert "dataset_bandgap_eV" in df.columns

def test_config_completeness_report():
    import subprocess
    res = subprocess.run([".venv/bin/python", "scripts/generate_pilot_config_report.py"], capture_output=True, text=True)
    assert res.returncode == 0
    with open("results/oracle_calibration_pilot/config_completeness.json") as f:
        rep = json.load(f)
    found_unknown = any(item["status"] == "DATASET_UNKNOWN" for item in rep)
    assert found_unknown

def test_export_blocked_structure(tmp_path):
    import subprocess
    res = subprocess.run([".venv/bin/python", "scripts/export_oracle_jobs.py", "--backend", "qe", "--subset", "calibration_pilot", "--dry-run"], capture_output=True, text=True)
    assert res.returncode == 0
    # check that STRUCTURE_BLOCKED.txt exists in the bundle
    bundle = Path("oracle_calibration_pilot_bundle")
    assert bundle.exists()
    qe_dir = bundle / "qe"
    has_blocked = False
    for job_dir in qe_dir.iterdir():
        if job_dir.is_dir() and (job_dir / "STRUCTURE_BLOCKED.txt").exists():
            has_blocked = True
            break
    assert has_blocked

