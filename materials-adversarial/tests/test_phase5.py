"""Tests for Phase 5 Deletion Transfer logic."""

import json
from pathlib import Path

import pytest
import pandas as pd
from rdkit import Chem

from materials_adv.data.tokenizer import tokenize


def test_deletion_bank_validity_and_constraints():
    """Verify that all generated deletion candidates satisfy the attack constraints."""
    repo_root = Path(__file__).resolve().parent.parent
    bank_path = repo_root / "results" / "candidate_banks" / "deletion_transfer_phase5" / "deletion_candidates.jsonl"
    
    if not bank_path.exists():
        pytest.skip("Deletion bank not generated yet.")
        
    with bank_path.open("r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
        
    assert len(recs) > 0
    
    for rec in recs:
        # Must be RDKit valid
        assert rec["valid"] is True
        
        orig_tokens = tokenize(rec["original_representation"])
        cand_tokens = tokenize(rec["candidate_representation"])
        
        # Exactly 1 token deleted
        assert len(cand_tokens) == len(orig_tokens) - 1
        
        # Verify chemistry-changing
        assert rec["canonical_original"] != rec["canonical_candidate"]


def test_deletion_evaluation_results():
    """Verify evaluation outputs matched expected directions."""
    repo_root = Path(__file__).resolve().parent.parent
    eval_path = repo_root / "results" / "phase5_deletion_transfer" / "deletion_evaluation.json"
    
    if not eval_path.exists():
        pytest.skip("Evaluation not run yet.")
        
    with eval_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
        
    models = report["models"]
    
    # 1. Stress Exceedance Rate
    base_sr = models["ordinary_baseline"]["stress_exceedance_rate"]
    ctrl_sr = models["architecture_control"]["stress_exceedance_rate"]
    rand_sr = models["randomization_robust"]["stress_exceedance_rate"]
    mix_sr = models["mixed_robust"]["stress_exceedance_rate"]
    
    # Robust models should have lower exceedance rate than baseline
    assert rand_sr < base_sr
    assert mix_sr < base_sr
    
    # 2. Mean Drift
    base_drift = models["ordinary_baseline"]["mean_drift"]
    rand_drift = models["randomization_robust"]["mean_drift"]
    
    assert rand_drift < base_drift

    # 3. Paired Significance
    # Check that randomization_robust significantly improved over baseline
    comparisons = report["paired_generalization"]
    rand_vs_base = next(c for c in comparisons if c["reference"] == "ordinary_baseline" and c["hypothesis"] == "randomization_robust")
    assert rand_vs_base["significant"] is True
    assert rand_vs_base["direction"] == "improved"
    
    # Check fixed threshold is 0.4862
    for m in models.values():
        assert abs(m["fixed_stress_threshold"] - 0.4862) < 1e-4

