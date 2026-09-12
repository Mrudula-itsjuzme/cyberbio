"""Tests for Phase 2 Paired Benchmark runner and candidate set integrity."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from materials_adv.experiments.pipeline import (
    AttackCandidate,
    compare_paired_records,
    score_candidate_bank,
)
from materials_adv.evaluation.attack_metrics import SuccessCriterion
from rdkit import Chem
from materials_adv.evaluation.records import AttackRecord


def test_compare_paired_records_raises_on_mismatched_ids():
    criterion = SuccessCriterion(min_abs_drift=0.4619, require_valid=True)
    
    rec_a = AttackRecord.from_dict({
        "attack_id": "c1",
        "sample_id": "s1",
        "original_representation": "[*]CC[*]",
        "adversarial_representation": "[*]CCC[*]",
        "attack_type": "insertion",
        "attack_budget": 1,
        "number_of_changes": 1,
        "validity_status": "valid",
        "plausibility_status": "plausible",
        "original_prediction": 1.0,
        "adversarial_prediction": 2.0,
        "signed_prediction_drift": 1.0,
        "absolute_prediction_drift": 1.0,
        "attack_success": True,
    })

    rec_b = AttackRecord.from_dict({
        "attack_id": "c2",
        "sample_id": "s1",
        "original_representation": "[*]CC[*]",
        "adversarial_representation": "[*]CCCC[*]",
        "attack_type": "insertion",
        "attack_budget": 1,
        "number_of_changes": 2,
        "validity_status": "valid",
        "plausibility_status": "plausible",
        "original_prediction": 1.0,
        "adversarial_prediction": 2.5,
        "signed_prediction_drift": 1.5,
        "absolute_prediction_drift": 1.5,
        "attack_success": True,
    })

    with pytest.raises(ValueError, match="paired comparison requires identical ordered candidate IDs"):
        compare_paired_records([rec_a], [rec_b], criterion=criterion)


def test_compare_paired_records_succeeds_on_matching_candidates():
    criterion = SuccessCriterion(min_abs_drift=0.4619, require_valid=True)

    rec_clean = AttackRecord.from_dict({
        "attack_id": "c1",
        "sample_id": "s1",
        "original_representation": "[*]CC[*]",
        "adversarial_representation": "[*]CCC[*]",
        "attack_type": "insertion",
        "attack_budget": 1,
        "number_of_changes": 1,
        "validity_status": "valid",
        "plausibility_status": "plausible",
        "original_prediction": 1.0,
        "adversarial_prediction": 2.0,
        "signed_prediction_drift": 1.0,
        "absolute_prediction_drift": 1.0,
        "attack_success": True,
    })

    rec_defended = AttackRecord.from_dict({
        "attack_id": "c1",
        "sample_id": "s1",
        "original_representation": "[*]CC[*]",
        "adversarial_representation": "[*]CCC[*]",
        "attack_type": "insertion",
        "attack_budget": 1,
        "number_of_changes": 1,
        "validity_status": "valid",
        "plausibility_status": "plausible",
        "original_prediction": 1.0,
        "adversarial_prediction": 1.2,
        "signed_prediction_drift": 0.2,
        "absolute_prediction_drift": 0.2,
        "attack_success": False,
    })

    res = compare_paired_records([rec_clean], [rec_defended], criterion=criterion)
    assert res["paired"] is True
    assert res["clean"]["insertion"]["n_success"] == 1
    assert res["defended"]["insertion"]["n_success"] == 0
    assert res["defended_minus_clean"]["insertion"]["success_rate"] == -1.0
