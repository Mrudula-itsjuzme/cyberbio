import pytest
import os
import json
import csv

def test_phase13_outputs():
    run_dir = "results/phase13_oracle_design/run_1"
    assert os.path.exists(run_dir)
    
    # 1. Edit count <= 3
    shortlist_path = os.path.join(run_dir, "candidate_shortlist.csv")
    assert os.path.exists(shortlist_path)
    
    with open(shortlist_path, "r") as f:
        reader = csv.DictReader(f)
        candidates = list(reader)
        assert len(candidates) <= 20
        for c in candidates:
            assert int(c["edit_count"]) <= 3
            assert int(c["edit_count"]) > 0
            
    # 2. Source/candidate graph distinction
    pairs_path = os.path.join(run_dir, "source_candidate_pairs.csv")
    assert os.path.exists(pairs_path)
    
    with open(pairs_path, "r") as f:
        reader = csv.DictReader(f)
        pairs = list(reader)
        for p in pairs:
            assert p["source_smiles"] != p["candidate_smiles"]
            assert int(p["edit_count"]) <= 3

    # 3. No source-label inheritance in candidate definition (No true labels passed to shortlist)
    with open(shortlist_path, "r") as f:
        content = f.read()
        assert "true_label" not in content
        assert "bandgap" not in content.lower()

    # 4. Deterministic generation
    assert os.path.exists(os.path.join(run_dir, "summary.json"))
    assert os.path.exists(os.path.join(run_dir, "reproducibility.json"))
    
    # 5. Candidate/source pairing integrity
    with open(pairs_path, "r") as f:
        reader = csv.DictReader(f)
        pairs = list(reader)
        assert len(pairs) == len(candidates)
