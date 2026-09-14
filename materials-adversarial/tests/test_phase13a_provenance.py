import os
import pytest
import pandas as pd
import json

RESULTS_DIR = "results/phase13a_provenance/run_1"

def test_provenance_lock_files_exist():
    assert os.path.exists(os.path.join(RESULTS_DIR, "revised_target_provenance.json"))
    assert os.path.exists(os.path.join(RESULTS_DIR, "candidate_shortlist_verified.csv"))

def test_provenance_lock_content():
    with open(os.path.join(RESULTS_DIR, "revised_target_provenance.json"), "r") as f:
        data = json.load(f)
    
    assert data["target_column"] == "bandgap_chain"
    assert data["dft_functional"] == "UNKNOWN"
    assert data["basis_set"] == "UNKNOWN"

def test_candidate_shortlist_verified():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "candidate_shortlist_verified.csv"))
    assert "graph_drift" in df.columns
    assert "transformer_drift" in df.columns
    assert not df["graph_drift"].isnull().any()
    assert not df["transformer_drift"].isnull().any()
    
    # Check that values are numeric, not "UNKNOWN"
    assert pd.api.types.is_numeric_dtype(df["graph_drift"])
    assert pd.api.types.is_numeric_dtype(df["transformer_drift"])

if __name__ == "__main__":
    pytest.main([__file__])
