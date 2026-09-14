import os
import pytest
import pandas as pd
import json

RESULTS_DIR = "results/phase14_oracle_feasibility/run_1"

def test_phase14_files_exist():
    assert os.path.exists(os.path.join(RESULTS_DIR, "target_definition_verified.json"))
    assert os.path.exists(os.path.join(RESULTS_DIR, "available_oracle_tools.json"))
    assert os.path.exists(os.path.join(RESULTS_DIR, "oracle_decision.json"))

def test_no_speculative_claims():
    with open(os.path.join(RESULTS_DIR, "target_definition_verified.json"), "r") as f:
        data = json.load(f)
    
    assert data["dft_functional"] == "UNKNOWN"
    assert data["basis_set"] == "UNKNOWN"

def test_oracle_decision():
    with open(os.path.join(RESULTS_DIR, "oracle_decision.json"), "r") as f:
        data = json.load(f)
    
    assert data["verdict"] == "NOT_FEASIBLE_IN_CURRENT_ENVIRONMENT"

def test_calibration_manifest():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "calibration_manifest_verified.csv"))
    assert len(df) == 20
    assert "smiles" in df.columns
    assert "bandgap_chain" in df.columns

def test_failures_logged():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "construction_failures.csv"))
    assert len(df) == 20
    assert "failure_category" in df.columns
    assert (df["failure_category"] == "software limitation").all()

if __name__ == "__main__":
    pytest.main([__file__])
