from pathlib import Path

import pandas as pd
from rdkit import Chem

from materials_adv.domain.chemistry.polymer_repeat import PolymerRepeatGraph
from scripts.prepare_physical_oracle_phase import pair_is_topologically_eligible


def test_repeat_graph_rejects_mismatched_attachment_bonds():
    graph = PolymerRepeatGraph("[*]=CC[*]")
    assert not graph.is_unambiguous
    assert graph.build_oligomer(2) is None


def test_pair_gate_rejects_parser_induced_hydrogenated_halogen():
    eligible, reason, elements = pair_is_topologically_eligible(
        "[*]C=C([*])c1cccc(C#C)c1",
        "[*]C=I([*])c1cccc(I#C)c1",
    )
    assert not eligible
    assert reason == "candidate_hydrogenated_halogen_rejected"
    assert elements == []


def test_prepared_oracle_phase_is_fail_closed_and_test_sealed():
    run = Path("results/physical_oracle_phase/20260921")
    calibration = pd.read_csv(run / "calibration_manifest.csv")
    structures = pd.read_csv(run / "structure_manifest.csv")
    pairs = pd.read_csv(run / "paired_pilot_manifest.csv")
    assert set(calibration["ml_split"]) <= {"train", "val"}
    assert set(calibration["attachment_point_count"]) == {2}
    assert set(calibration["formal_charge"]) == {0}
    assert structures["structure_sha256"].notna().all()
    assert not structures["elements"].str.contains(r"\*").any()
    assert pairs["oracle_status"].eq("BLOCKED_PENDING_CALIBRATION").all()
    assert pairs["delta_T_eV"].isna().all()
    assert len(pairs) == 4
    assert len(set(zip(pairs["attack_family"], pairs["target_model"]))) == 4
    convergence = pd.read_csv(run / "convergence_plan.csv")
    assert set(convergence["gap_stratum"]) == {"low", "mid", "high"}
    assert len(convergence) == 30


def test_required_elements_are_symbols_not_numeric_targets():
    import json

    run = Path("results/physical_oracle_phase/20260921")
    elements = json.loads((run / "required_elements.json").read_text())
    assert elements
    assert all(Chem.GetPeriodicTable().GetAtomicNumber(symbol) > 0 for symbol in elements)
    assert not any(any(character.isdigit() for character in symbol) for symbol in elements)
