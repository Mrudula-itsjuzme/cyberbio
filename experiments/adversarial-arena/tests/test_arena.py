from pathlib import Path
import json

from arena.adaptive import ThompsonController
from arena.buffers import HardExampleBuffer
from arena.contracts import CandidateRecord, SourceSample
from arena.defenders import RidgeSmokeDefender
from arena.metrics import attack_metrics, gain


def candidate(drift=1.0, status="valid", family="RandomSMILES"):
    return CandidateRecord("s","CC","C(C)",family,family,0,validity_status=status,
        target_query_count=2,drift=drift,true_target=2.0)

def test_buffer_selects_valid_top_k_and_preserves_round():
    buffer=HardExampleBuffer()
    chosen=buffer.add_top_k([candidate(.2),candidate(2),candidate(9,"invalid")],1,3)
    assert [x.drift for x in chosen]==[2]
    assert chosen[0].metadata["selected_round"]==3

def test_metrics_do_not_count_invalid_drift():
    result=attack_metrics([candidate(2),candidate(100,"invalid")])
    assert result["mean_drift"]==2
    assert result["validity_rate"]==.5
    assert result["queries"]==4

def test_thompson_state_round_trip_evidence():
    controller=ThompsonController(["a","b"],7)
    selected=controller.select(); controller.update(selected,.75)
    state=controller.state()
    assert state["alpha"][selected]==1.75
    assert len(state["history"])==1

def test_defender_really_updates_and_checkpoints(tmp_path):
    defender=RidgeSmokeDefender(4,dimension=16)
    clean=[SourceSample("a","CC",1.0),SourceSample("b","NN",3.0)]
    defender.train(clean,[],round_id=0); before=defender.weights.copy()
    defender.train(clean,[candidate(3)],round_id=1)
    assert (before!=defender.weights).any()
    path=tmp_path/"checkpoint.json"; defender.checkpoint(str(path))
    restored=RidgeSmokeDefender(0,dimension=16); restored.load_checkpoint(str(path))
    assert restored.predict(["CC"])==defender.predict(["CC"])

def test_gain_ratio_definition():
    assert gain(4,2)==2
    assert gain(4,0) is None

def test_smoke_artifacts_are_complete():
    root=Path(__file__).parents[1]/"results"/"arena"/"smoke_seed20260920_a"
    summary=json.loads((root/"run_summary.json").read_text())
    assert summary["status"]=="RUN_COMPLETE"
    assert len(summary["rounds"])==2
    for index in range(2):
        round_dir=root/f"round_{index:03d}"
        assert (round_dir/"COMPLETE.json").exists()
        assert (round_dir/"defender_checkpoint.json").exists()

