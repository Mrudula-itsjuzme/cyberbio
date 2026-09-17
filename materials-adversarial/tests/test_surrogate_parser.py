import os
import pytest
from materials_adv.oracle.surrogate_parser import SurrogateParser
from pathlib import Path

@pytest.fixture
def parser():
    return SurrogateParser()

@pytest.fixture
def mock_outputs(tmp_path):
    success_file = tmp_path / "success.out"
    success_file.write_text("""
     Program PWSCF v.6.8 starts on 14Sep2026 at 10:00:00 
     !    total energy              =     -123.45678900 Ry
     highest occupied, lowest unoccupied level (ev):     -5.0000    -2.0000
     JOB DONE.
    """)
    
    metallic_file = tmp_path / "metallic.out"
    metallic_file.write_text("""
     Program PWSCF v.6.8 starts on 14Sep2026 at 10:00:00 
     !    total energy              =     -100.00000000 Ry
     highest occupied level (ev):     -4.5000
     JOB DONE.
    """)
    
    unconverged_file = tmp_path / "unconverged.out"
    unconverged_file.write_text("""
     Program PWSCF v.6.8 starts on 14Sep2026 at 10:00:00 
     convergence NOT achieved after 100 iterations: stopping
    """)
    
    missing_file = tmp_path / "missing.out"
    missing_file.write_text("""
     Program PWSCF v.6.8 starts on 14Sep2026 at 10:00:00 
     !    total energy              =     -123.45678900 Ry
     JOB DONE.
    """)
    
    crash_file = tmp_path / "crash.out"
    crash_file.write_text("""
     Program PWSCF v.6.8 starts on 14Sep2026 at 10:00:00 
     %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     Error in routine read_namelists (1):
     reading namelist system
     %^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     CRASH
    """)
    
    malformed_file = tmp_path / "malformed.out"
    malformed_file.write_text("""
     Program PWSCF v.6.8 starts on 14Sep2026 at 10:00:00 
     Some random text
    """)
    
    return {
        "success": str(success_file),
        "metallic": str(metallic_file),
        "unconverged": str(unconverged_file),
        "missing": str(missing_file),
        "crash": str(crash_file),
        "malformed": str(malformed_file)
    }

def test_success_parser(parser, mock_outputs):
    res = parser.parse_qe_output(mock_outputs["success"])
    assert res["status"] == "SUCCESS"
    assert res["converged"] is True
    assert res["HOMO_or_VBM_eV"] == -5.0
    assert res["LUMO_or_CBM_eV"] == -2.0
    assert res["surrogate_gap_eV"] == 3.0
    assert res["total_energy_eV"] == pytest.approx(-123.45678900 * 13.6056980659)
    
def test_metallic_parser(parser, mock_outputs):
    res = parser.parse_qe_output(mock_outputs["metallic"])
    assert res["status"] == "METALLIC"
    assert res["converged"] is True
    assert res["surrogate_gap_eV"] == 0.0
    assert "System is metallic / zero-gap" in res["warnings"]

def test_unconverged_parser(parser, mock_outputs):
    res = parser.parse_qe_output(mock_outputs["unconverged"])
    assert res["status"] == "FAILED"
    assert res["converged"] is False
    assert res["failure_reason"] == "SCF nonconvergence"

def test_missing_levels(parser, mock_outputs):
    res = parser.parse_qe_output(mock_outputs["missing"])
    assert res["status"] == "FAILED"
    assert res["failure_reason"] == "Missing levels in output"

def test_crash_parser(parser, mock_outputs):
    res = parser.parse_qe_output(mock_outputs["crash"])
    assert res["status"] == "FAILED"
    assert res["failure_reason"] == "Backend crash or error"

def test_malformed_parser(parser, mock_outputs):
    res = parser.parse_qe_output(mock_outputs["malformed"])
    assert res["status"] == "FAILED"
    assert res["failure_reason"] == "Malformed output or incomplete job"
