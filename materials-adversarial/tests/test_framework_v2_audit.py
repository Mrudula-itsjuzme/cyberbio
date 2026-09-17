"""Regression tests for the Framework-V2 benchmark forensic audit.

See docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md. These tests pin both directions:

* the canonical facts the framework MUST reproduce (predictor equivalence, validation
  MAE, unit/scaling, edit budget, validity);
* the three protocol defects found in the developmental run (out-of-split sources,
  hash-stub predictor, query-accounting under-count), so they cannot silently return
  as if they were results.

Model-backed tests SKIP when the canonical GraphMPNN checkpoint is absent from the
working copy, so the suite still runs on a checkout without `results/`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from materials_adv.framework.budget import Budget
from materials_adv.framework.interfaces import Candidate
from materials_adv.framework.objectives import TargetIncrease, UntargetedDrift
from materials_adv.framework.predictors import GraphMPNNPredictor, HashStubPredictor

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_module("framework_v2_audit", "scripts/audit_framework_v2_benchmark.py")

CKPT = ROOT / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
SCALER = ROOT / "results/models/transformer_regressor/scaler.json"
MANIFEST = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
DEV_SOURCES = ROOT / "results/developmental_run_1/benchmark_sources.csv"

requires_model = pytest.mark.skipif(
    not (CKPT.exists() and SCALER.exists()),
    reason="canonical GraphMPNN checkpoint/scaler not present in this checkout",
)


@pytest.fixture(scope="module")
def predictor() -> GraphMPNNPredictor:
    return GraphMPNNPredictor(CKPT, SCALER)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Canonical model / scaler integrity
# ---------------------------------------------------------------------------
def test_checkpoint_and_scaler_match_release_manifest() -> None:
    if not CKPT.exists() or not SCALER.exists():
        pytest.skip("canonical artefacts not present")
    assert _sha256(CKPT) == MANIFEST["hashes"]["GraphMPNN_Small_ckpt"]
    assert _sha256(SCALER) == MANIFEST["hashes"]["scaler"]


@requires_model
def test_graph_predictor_has_canonical_parameter_count(predictor: GraphMPNNPredictor) -> None:
    assert predictor.parameter_count() == 27585


@requires_model
def test_predictor_wrapper_equals_canonical_phase11c_evaluator(predictor: GraphMPNNPredictor) -> None:
    """V2 predictor == canonical GraphMPNN predictor, bit for bit."""
    p11c = _load_module("p11c_eval", "scripts/run_phase11c_canonical_verification.py")
    val_df = audit.DF.iloc[audit.VAL_IDX].copy().reset_index(drop=True).head(20)

    canonical = p11c.eval_model_canonical(
        predictor.model, val_df, None, predictor.scaler, predictor.device, is_graph=True
    )["Predictions"]
    wrapper = predictor.predict_batch(val_df["original_representation"].tolist())

    assert np.max(np.abs(np.asarray(canonical) - np.asarray(wrapper))) == 0.0


@requires_model
def test_hash_stub_predictor_is_not_the_canonical_model(predictor: GraphMPNNPredictor) -> None:
    """The developmental 'predictor' must never again be mistaken for the model."""
    smiles = audit.DF.iloc[audit.VAL_IDX].head(20)["original_representation"].tolist()
    stub = HashStubPredictor()
    diff = np.abs(np.asarray(predictor.predict_batch(smiles)) - np.asarray(stub.predict_batch(smiles)))
    assert diff.max() > 1.0
    # the stub is uniform on [0, 10) -- impossible for a real bandgap prediction
    assert all(0.0 <= v < 10.0 for v in stub.predict_batch(smiles))


@requires_model
def test_validation_mae_reproduces_canonical_0_41119(predictor: GraphMPNNPredictor) -> None:
    val_df = audit.DF.iloc[audit.VAL_IDX].copy().reset_index(drop=True)
    preds = np.asarray(predictor.predict_batch(val_df["original_representation"].tolist()))
    truth = val_df["property_value"].astype(float).to_numpy()
    mae = float(np.mean(np.abs(preds - truth)))
    assert abs(mae - 0.41119) < 0.005


# ---------------------------------------------------------------------------
# Units / scaling
# ---------------------------------------------------------------------------
@requires_model
def test_scaler_is_applied_exactly_once(predictor: GraphMPNNPredictor) -> None:
    import torch

    src = audit.DF.iloc[audit.VAL_IDX[0]]["original_representation"]
    cands = audit.AtomSubstitutionAttack().apply(Candidate(src, provenance=["source"]))
    cand = cands[0].identifier

    x, adj, mask = audit.raw_graph_batch([src, cand])
    with torch.no_grad():
        raw = predictor.model(x, adj, mask).view(-1).numpy()
    assert np.all(np.isfinite(raw))

    src_ev = float(predictor.scaler.inverse_transform(np.array([[raw[0]]]))[0][0])
    cand_ev = float(predictor.scaler.inverse_transform(np.array([[raw[1]]]))[0][0])

    # library path agrees with one inverse transform, to float32 precision
    assert abs(predictor.predict(src) - src_ev) < 1e-4
    assert abs(predictor.predict(cand) - cand_ev) < 1e-4

    # a double inverse transform would change the number -> one is correct
    src_double = float(predictor.scaler.inverse_transform(np.array([[src_ev]]))[0][0])
    assert abs(src_double - src_ev) > 1.0


@requires_model
def test_drift_is_measured_in_eV(predictor: GraphMPNNPredictor) -> None:
    smiles = audit.DF.iloc[audit.VAL_IDX].head(8)["original_representation"].tolist()
    preds = np.asarray(predictor.predict_batch(smiles))
    assert preds.min() > 0.0 and preds.max() < 12.0  # bandgaps in eV
    drift_ev = abs(predictor.predict(smiles[0]) - predictor.predict(smiles[1]))
    assert drift_ev == pytest.approx(float(abs(preds[0] - preds[1])), abs=1e-9)
    # subtracting in scaled space gives a different, unitless number
    scaled = [predictor.scaler.transform(p) for p in (preds[0], preds[1])]
    assert abs(abs(scaled[0] - scaled[1]) - drift_ev) > 1e-6


# ---------------------------------------------------------------------------
# Source split
# ---------------------------------------------------------------------------
def test_rebuilt_sources_are_validation_only() -> None:
    sources = audit.select_sources(audit.VAL_IDX)
    assert len(sources) == 30
    train, val, test = set(audit.TRAIN_IDX), set(audit.VAL_IDX), set(audit.TEST_IDX)
    for idx, _ in sources:
        assert idx in val
        assert idx not in train and idx not in test


def test_developmental_sources_were_out_of_split() -> None:
    """Pins the frozen defect: 25 train + 5 test, 0 validation (audit S1)."""
    if not DEV_SOURCES.exists():
        pytest.skip("developmental artefact not present")
    recorded = pd.read_csv(DEV_SOURCES)["smiles"].tolist()
    lookup = {s: i for i, s in enumerate(audit.DF["original_representation"].tolist())}
    train, val, test = set(audit.TRAIN_IDX), set(audit.VAL_IDX), set(audit.TEST_IDX)
    memberships = []
    for smi in recorded:
        idx = lookup[smi]
        memberships.append(
            "train" if idx in train else "test" if idx in test else "val" if idx in val else "ABSENT"
        )
    assert memberships.count("train") == 25
    assert memberships.count("test") == 5
    assert memberships.count("val") == 0


def test_developmental_sources_fingerprint_full_dataset_shuffle() -> None:
    """The recorded list is exactly `shuffle(range(len(df)))` with seed 42."""
    if not DEV_SOURCES.exists():
        pytest.skip("developmental artefact not present")
    recorded = pd.read_csv(DEV_SOURCES)["smiles"].tolist()
    reproduced = [smi for _, smi in audit.reproduce_developmental_sources()]
    assert reproduced == recorded


# ---------------------------------------------------------------------------
# Edit budget
# ---------------------------------------------------------------------------
def test_simple_substitution_respects_three_edit_budget() -> None:
    for idx, smi in audit.select_sources(audit.VAL_IDX)[:6]:
        for cand in audit.AtomSubstitutionAttack().apply(Candidate(smi, provenance=["source"])):
            info = audit.inspect_candidate(smi, cand.identifier)
            assert info["independent_edit_count"] <= 3
            assert info["element_change_count"] <= 3


def test_provenance_counter_is_not_an_edit_distance() -> None:
    """Budget.validate_edit_distance counts operator applications, not chemistry.

    A two-application chain that re-types the same site (C -> N -> O) nets ONE
    element change, so the stored count and the graph count disagree.
    """
    src = "[*]CCCCCC[*]"
    c1 = Candidate(src, provenance=["source", "subst"])
    c2 = Candidate(src, provenance=["source", "subst", "subst"])
    budget = Budget(max_queries=10, max_edits=3)
    assert budget.validate_edit_distance(c1, Candidate(src, provenance=["source"]))
    assert budget.validate_edit_distance(c2, Candidate(src, provenance=["source"]))

    changed_once = audit.independent_graph_edits(
        audit.Chem.MolFromSmiles(src), audit.Chem.MolFromSmiles("[*]NCCCCC[*]")
    )
    assert changed_once["independent_edit_count"] == 1


def test_string_distance_would_falsely_report_edit_creep() -> None:
    """Canonicalisation inflates SMILES distance; it is not an edit measure."""
    distances, independent = [], []
    for idx, src in audit.select_sources(audit.VAL_IDX)[:6]:
        for cand in audit.AtomSubstitutionAttack().apply(Candidate(src, provenance=["source"])):
            info = audit.inspect_candidate(src, cand.identifier)
            distances.append(info["string_edit_distance"])
            independent.append(info["independent_edit_count"])
    assert distances, "expected one-edit candidates"
    assert set(independent) == {1}
    # a SINGLE atom substitution can score 10+ on string distance after canonicalisation
    assert max(distances) > 3


# ---------------------------------------------------------------------------
# Query accounting
# ---------------------------------------------------------------------------
class _RecordingPredictor:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.calls = 0

    def predict(self, representation) -> float:
        self.calls += 1
        return self.value


def test_objective_spends_two_predictor_calls_per_query() -> None:
    """Pins the 2x under-count: one budget query, two model asks."""
    pred = _RecordingPredictor()
    budget = Budget(max_queries=3, max_edits=3)
    src = Candidate("A", provenance=["source"])
    cand = Candidate("B", provenance=["source", "subst"])
    objective = UntargetedDrift()

    for _ in range(budget.max_queries):
        assert budget.use_query()
        objective.evaluate(pred, src, cand)

    assert budget.queries_used == 3
    assert pred.calls == 6
    assert pred.calls > budget.queries_used


def test_budget_never_over_counts() -> None:
    budget = Budget(max_queries=2, max_edits=3)
    assert budget.use_query() and budget.use_query()
    assert not budget.use_query()
    assert budget.is_exhausted()
    assert budget.queries_used == 2


@requires_model
def test_budgeted_predictor_exposes_forward_call_accounting(predictor: GraphMPNNPredictor) -> None:
    before = predictor.forward_calls
    predictor.predict_batch(["CCO", "CCN"])
    assert predictor.forward_calls > before
    assert predictor.molecules_scored >= 2


# ---------------------------------------------------------------------------
# Candidate validity
# ---------------------------------------------------------------------------
def test_candidate_validity_gate() -> None:
    for idx, smi in audit.select_sources(audit.VAL_IDX)[:6]:
        for cand in audit.AtomSubstitutionAttack().apply(Candidate(smi, provenance=["source"]))[:5]:
            info = audit.inspect_candidate(smi, cand.identifier)
            assert info["parse_ok"] and info["sanitize_ok"] and info["valence_ok"]
            assert info["components"] == 1
            assert info["wildcards_candidate"] == info["wildcards_source"]
            assert info["differs_from_source"]
            assert info["valid"]


def test_attachment_semantics_are_preserved() -> None:
    idx, smi = audit.select_sources(audit.VAL_IDX)[0]
    for cand in audit.AtomSubstitutionAttack().apply(Candidate(smi, provenance=["source"]))[:5]:
        info = audit.inspect_candidate(smi, cand.identifier)
        assert info["wildcards_source"] >= 1
        assert info["attachment_valid"] is True


# ---------------------------------------------------------------------------
# Operator semantics
# ---------------------------------------------------------------------------
def test_functional_group_operator_is_atom_level_not_motif_swap() -> None:
    """FunctionalGroupReplacementAttack only ever replaced aliphatic carbons."""
    op = audit.FunctionalGroupReplacementAttack()
    assert op.motifs
    patterns = {p for p, _ in op.motifs}
    assert patterns == {"[C;X4;h1,h2,h3]"}

    idx, smi = audit.select_sources(audit.VAL_IDX)[0]
    for cand in op.apply(Candidate(smi, provenance=["source"]))[:5]:
        info = audit.inspect_candidate(smi, cand.identifier)
        assert info["components"] == 1  # no disconnected fragments
        assert info["atom_delta"] <= 3


def test_phase12b_and_v2_operators_are_different_threat_models() -> None:
    assert "Cl" in audit.PHASE12B_ALLOWED_TOKENS
    assert "Cl" not in audit.AtomSubstitutionAttack().atoms
    # Phase 12B cannot reach aromatic atoms; V2 can.
    idx, smi = audit.select_sources(audit.VAL_IDX)[0]
    sites = [
        audit.identify_substitution(smi, c.identifier)
        for c in audit.AtomSubstitutionAttack().apply(Candidate(smi, provenance=["source"]))
    ]
    assert sites and all(s["matched"] for s in sites)


def test_directional_success_is_sign_only() -> None:
    """TARGET_INCREASE 'success' is a direction test, not an attack-success metric."""
    class _Map:
        def __init__(self, values):
            self.values = values

        def predict(self, c):
            return self.values[c.identifier]

    pred = _Map({"src": 5.0, "tiny": 5.0001, "big": 9.0})
    src = Candidate("src")
    objective = TargetIncrease()
    assert objective.evaluate(pred, src, Candidate("tiny")) > 0
    # a physically meaningless 1e-4 eV shift counts exactly as much as a 4 eV shift
    assert objective.evaluate(pred, src, Candidate("big")) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Framework isolation
# ---------------------------------------------------------------------------
GENERIC_MODULES = [
    "materials_adv.framework.interfaces",
    "materials_adv.framework.objectives",
    "materials_adv.framework.budget",
    "materials_adv.framework.accounting",
    "materials_adv.framework.search",
    "materials_adv.attacks.search.evolutionary",
    "materials_adv.attacks.search.llm.proposer",
]


def _rdkit_in_sys_modules(modules: list[str]) -> str:
    code = "import sys\n" + "\n".join(f"import {m}" for m in modules) + "\nprint('rdkit' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_generic_framework_and_search_do_not_import_rdkit() -> None:
    """The generic abstractions and the search layer must be domain-agnostic.

    The audit found the 'generic' evolutionary search importing RDKit at import time and
    defaulting to a chemistry validity checker; this now imports nothing chemical.
    ``materials_adv.framework.predictors`` is deliberately excluded: it is the binding
    to the materials model stack and therefore reaches the graph featuriser.
    """
    assert _rdkit_in_sys_modules(GENERIC_MODULES) == "False"


def test_no_module_in_the_generic_layers_imports_rdkit_in_source() -> None:
    """Source-level guard: no direct RDKit import anywhere in the generic layers."""
    offenders = []
    for layer in (ROOT / "src/materials_adv/framework", ROOT / "src/materials_adv/attacks/search"):
        for path in layer.rglob("*.py"):
            text = path.read_text()
            if "import rdkit" in text or "from rdkit" in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_audit_top20_table_schema() -> None:
    top20 = ROOT / "results/framework_v2/audit_top20_candidates.csv"
    if not top20.exists():
        pytest.skip("audit table not present")
    df = pd.read_csv(top20)
    required = {
        "source_id", "source_representation", "candidate_representation", "operator",
        "strategy", "edit_count", "independent_edit_count", "source_prediction_eV",
        "candidate_prediction_eV", "absolute_drift_eV", "source_atom_count",
        "candidate_atom_count", "formal_charge_delta", "valid", "attachment_valid",
        "prediction_range_class",
    }
    assert required.issubset(df.columns)
    assert len(df) == 20
    assert (df["independent_edit_count"] <= 3).all()
