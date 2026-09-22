import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/physical_oracle_phase/20260922"
PREP = ROOT / "results/physical_oracle_phase/20260921"


def test_failed_convergence_cannot_promote_protocol_or_downstream_execution():
    audit = json.loads((RUN / "pre_run_audit.json").read_text())
    convergence = json.loads((RUN / "convergence_manifest.json").read_text())
    assert audit["execution_decision"]["final_status"] == "CONVERGENCE_FAILED"
    assert audit["execution_decision"]["new_qe_jobs_launched"] == 0
    assert not audit["execution_decision"]["calibration_allowed"]
    assert not audit["execution_decision"]["attack_pairs_allowed"]
    assert not convergence["protocol_frozen"]
    assert not (RUN / "oracle_protocol_frozen.json").exists()
    assert not (RUN / "oracle_protocol.sha256").exists()


def test_frozen_pair_identities_and_signed_predictions_remain_exact():
    with (PREP / "paired_pilot_manifest.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    with (PREP / "model_predictions/signed_model_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    expected = [("18", "greedy", "GraphMPNN"), ("41", "greedy", "Transformer"),
                ("23", "metropolis", "GraphMPNN"), ("0", "metropolis", "Transformer")]
    assert [(row["source_id"], row["attack_family"], row["target_model"]) for row in pairs] == expected
    assert [(row["source_id"], row["attack_family"], row["target_model"]) for row in predictions] == expected
    assert all(float(row["delta_G_signed_eV"]) < 0 for row in predictions)


def test_not_run_outputs_use_missing_values_not_fake_zeroes():
    calibration = json.loads((RUN / "calibration_summary.json").read_text())
    attacks = json.loads((RUN / "attack_pair_oracle_summary.json").read_text())
    assert calibration["status"] == "NOT_RUN_CONVERGENCE_FAILED"
    assert calibration["metrics"] is None
    assert calibration["calibration_verdict"] is None
    assert attacks["status"] == "NOT_RUN_CONVERGENCE_FAILED"
    assert attacks["pairs_executed"] == 0
