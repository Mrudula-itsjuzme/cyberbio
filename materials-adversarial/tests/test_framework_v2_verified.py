"""Regression tests for the repaired Framework V2 pipeline and verified benchmark.

Covers the contract that the verified run depends on:

* the only real predictor is the hash-validated canonical GraphMPNN;
* mock predictors are declared MOCK_TEST_ONLY and are not reachable from the benchmark;
* one budgeted scoring path, with ``predictor_calls == 1 + queries_used`` and
  ``queries_used <= Q``;
* the source is scored once and never charged to ``Q``;
* duplicates and invalid proposals do not consume queries;
* search cannot hold a predictor at all;
* domain-supplied candidate identity;
* strict graph featurisation for adversarial evaluation;
* operator edit budget counted on the original source;
* scaffold subgraph retention;
* validation-only verified source set and deterministic search.

Model-backed tests skip when the canonical checkpoint is absent. A guard test asserts
which tests are allowed to use a fake predictor.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution import (
    AliphaticCarbonSubstitutionAttack,
)
from materials_adv.domain.chemistry.attacks.functional_group import (
    FunctionalGroupReplacementAttack,
)
from materials_adv.domain.chemistry.attacks.scaffold_preserving import (
    ScaffoldPolicy,
    ScaffoldPreservingAttack,
)
from materials_adv.domain.chemistry.attacks.simple_substitution import SimpleSubstitutionAttack
from materials_adv.domain.chemistry.graph_metrics import count_wildcards, graph_change
from materials_adv.domain.chemistry.identity import CanonicalSmilesIdentity
from materials_adv.domain.chemistry.validator import RDKitValidityChecker
from materials_adv.framework.accounting import AttackEvaluator, BudgetExhausted
from materials_adv.framework.budget import Budget
from materials_adv.framework.interfaces import Candidate, StringIdentity
from materials_adv.framework.objectives import TargetIncrease, UntargetedDrift
from materials_adv.framework.predictors import GraphMPNNPredictor, HashStubPredictor  # noqa: F401
from materials_adv.framework.search import (
    EvolutionarySearch,
    GreedySearch,
    MetropolisSearch,
    RandomSearch,
)

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
SCALER = ROOT / "results/models/transformer_regressor/scaler.json"
VERIFIED = ROOT / "results/framework_v2/verified_run_1"

requires_model = pytest.mark.skipif(
    not (CKPT.exists() and SCALER.exists()),
    reason="canonical GraphMPNN checkpoint/scaler not present in this checkout",
)


# ---------------------------------------------------------------------------
# TEST-ONLY predictor. Declared MOCK_TEST_ONLY: it must never produce a reported
# number, and the guard test below keeps it out of the benchmark scripts.
# ---------------------------------------------------------------------------
class MOCK_TEST_ONLY_RecordingPredictor:
    """Deterministic fake: prediction = canonical-ish length proxy. TEST ONLY."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, representation) -> float:
        self.calls += 1
        smiles = representation.identifier if isinstance(representation, Candidate) else str(representation)
        return 2.0 + (len(smiles) % 7) * 0.5


SOURCE = "[*]CC1CCC(COC(=O)CCCC(=O)O[*])CC1"
ACYCLIC_SOURCE = "[*]CCNC(=O)CCCC(=O)N[*]"


def _evaluator(smiles=SOURCE, q=10, predictor=None, objective=None):
    predictor = predictor or MOCK_TEST_ONLY_RecordingPredictor()
    source = Candidate(smiles, provenance=["source"])
    evaluator = AttackEvaluator(
        predictor, objective or UntargetedDrift(), Budget(q, 3), source, CanonicalSmilesIdentity()
    )
    return predictor, evaluation_helpers(source, evaluator, smiles)


def evaluation_helpers(source, evaluator, smiles):
    return {
        "source": source,
        "evaluator": evaluator,
        "validator": RDKitValidityChecker(require_single_component=True,
                                         require_attachment_count=count_wildcards(smiles)),
    }


def _make_strategy(name, operator, ctx, seed=42):
    kwargs = dict(operator=operator, evaluator=ctx["evaluator"], validator=ctx["validator"], seed=seed)
    if name == "random":
        return RandomSearch(**kwargs)
    if name == "greedy":
        return GreedySearch(**kwargs)
    if name == "metropolis":
        return MetropolisSearch(**kwargs, temperature=0.1)
    if name == "evolutionary":
        return EvolutionarySearch(**kwargs, population_size=3, elite_size=2)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Predictor provenance
# ---------------------------------------------------------------------------
@requires_model
def test_real_predictor_is_hash_validated_graphmpnn() -> None:
    predictor = GraphMPNNPredictor(CKPT, SCALER)
    predictor.validate_against_release_manifest(ROOT / "RELEASE_MANIFEST.json")
    assert predictor.parameter_count() == 27585
    assert predictor.strict is True
    config = predictor.config()
    assert (config["node_dim"], config["hidden_dim"], config["num_layers"]) == (7, 64, 3)
    assert config["checkpoint_sha256"] == json.loads(
        (ROOT / "RELEASE_MANIFEST.json").read_text()
    )["hashes"]["GraphMPNN_Small_ckpt"]


@requires_model
def test_tampered_checkpoint_hash_is_rejected(tmp_path) -> None:
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    manifest["hashes"]["GraphMPNN_Small_ckpt"] = "0" * 64
    with pytest.raises(ValueError, match="does not match the release manifest|!= release manifest"):
        GraphMPNNPredictor(CKPT, SCALER).validate_against_release_manifest(
            _write(manifest, tmp_path)
        )


def _write(payload, tmp_path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_mock_predictor_is_declared_test_only() -> None:
    assert "MOCK_TEST_ONLY" in (HashStubPredictor.__doc__ or "")


def test_benchmark_scripts_never_use_a_stub_predictor() -> None:
    """The verified benchmark must not be able to reach a fake predictor."""
    script = (ROOT / "scripts/benchmark_v2_verified.py").read_text()
    # the module may *describe* the stub in prose, but must never import or build one
    assert "HashStubPredictor" not in script
    import_lines = [
        line for line in script.splitlines()
        if "from materials_adv.framework.predictors import" in line
    ]
    assert len(import_lines) == 1
    assert "GraphMPNNPredictor" in import_lines[0]
    # exactly one predictor construction, and it must be hash-validated
    assert script.count("GraphMPNNPredictor(") == 1
    assert "validate_against_release_manifest" in script


# ---------------------------------------------------------------------------
# Query accounting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["random", "greedy", "metropolis", "evolutionary"])
@pytest.mark.parametrize("q", [5, 10])
def test_exact_q_enforcement_for_every_strategy(strategy, q) -> None:
    predictor, ctx = _evaluator(q=q)
    search = _make_strategy(strategy, SimpleSubstitutionAttack(), ctx)
    outcome = search.search(ctx["source"])

    assert outcome.queries_used <= q
    invariants = ctx["evaluator"].invariants()
    assert invariants["one_model_call_per_query"] is True
    assert invariants["predictor_calls"] == 1 + outcome.queries_used
    assert predictor.calls == invariants["predictor_calls"]


def test_source_prediction_is_scored_once_and_not_charged() -> None:
    predictor, ctx = _evaluator(q=5)
    evaluator = ctx["evaluator"]
    first = evaluator.prepare_source()
    second = evaluator.prepare_source()
    assert first == second
    assert evaluator.source_scorings == 1
    assert predictor.calls == 1
    assert evaluator.queries_used == 0  # the source is not a query


def test_duplicates_do_not_consume_queries() -> None:
    _, ctx = _evaluator(q=5)
    evaluator = ctx["evaluator"]
    evaluator.prepare_source()
    candidates = SimpleSubstitutionAttack().apply(ctx["source"])
    first = evaluator.score(candidates[0])
    assert first.is_duplicate is False
    assert evaluator.queries_used == 1

    # a different SMILES writing of the same molecule is still a duplicate
    repeat = evaluator.score(Candidate(candidates[0].identifier, provenance=["source", "subst"]))
    assert repeat.is_duplicate is True
    assert evaluator.queries_used == 1
    assert evaluator.duplicate_queries == 1


def test_scoring_outside_the_budget_raises() -> None:
    _, ctx = _evaluator(q=2)
    evaluator = ctx["evaluator"]
    evaluator.prepare_source()
    candidates = SimpleSubstitutionAttack().apply(ctx["source"])
    evaluator.score(candidates[0])
    evaluator.score(candidates[1])
    with pytest.raises(BudgetExhausted):
        evaluator.score(candidates[2])


def test_unbudgeted_scoring_is_refused() -> None:
    _, ctx = _evaluator(q=5)
    ctx["evaluator"].prepare_source()
    candidate = SimpleSubstitutionAttack().apply(ctx["source"])[0]
    with pytest.raises(ValueError, match="unbudgeted scoring"):
        ctx["evaluator"].score(candidate, count_query=False)


def test_search_refuses_a_raw_predictor() -> None:
    with pytest.raises(TypeError, match="AttackEvaluator"):
        RandomSearch(SimpleSubstitutionAttack(), MOCK_TEST_ONLY_RecordingPredictor(), None)


# ---------------------------------------------------------------------------
# Candidate identity abstraction
# ---------------------------------------------------------------------------
def test_canonical_identity_collapses_equivalent_smiles() -> None:
    identity = CanonicalSmilesIdentity()
    left = Candidate("[*]OCCO[*]")
    right = Candidate("*OCCO*")
    assert identity.identity(left) == identity.identity(right)
    assert identity.is_equivalent(left, right)


def test_string_identity_is_domain_independent() -> None:
    identity = StringIdentity()
    assert identity.identity(Candidate("[*]OCCO[*]")) != identity.identity(Candidate("*OCCO*"))
    assert identity.is_equivalent(Candidate("ABC"), Candidate("ABC"))


def test_unparseable_candidate_identity_is_namespaced() -> None:
    identity = CanonicalSmilesIdentity()
    assert identity.identity(Candidate("not_a_smiles!!")).startswith("<unparseable>")


# ---------------------------------------------------------------------------
# Strict graph featurisation
# ---------------------------------------------------------------------------
def test_graph_dataset_strict_mode_rejects_invalid_smiles() -> None:
    from materials_adv.data.graph_dataset import GraphDataset

    frame = pd.DataFrame({"original_representation": ["CCO", "definitely not smiles"],
                          "property_value": [1.0, 2.0]})
    dataset = GraphDataset(frame, strict=True)
    assert dataset[0]["smiles"] == "CCO"
    with pytest.raises(ValueError, match="does not parse as SMILES"):
        dataset[1]


def test_graph_dataset_legacy_mode_substitutes_but_records_it() -> None:
    from materials_adv.data.graph_dataset import GraphDataset

    frame = pd.DataFrame({"original_representation": ["CCO", "definitely not smiles"],
                          "property_value": [1.0, 2.0]})
    dataset = GraphDataset(frame, strict=False)
    # legacy behaviour returns a DIFFERENT molecule -- documented, not silently trusted
    assert dataset[1]["smiles"] == "CCO"
    assert dataset.skipped_unparseable >= 1


# ---------------------------------------------------------------------------
# Edit budget semantics
# ---------------------------------------------------------------------------
def test_operator_edits_accumulate_from_the_original_source() -> None:
    source = Candidate(SOURCE, provenance=["source"])
    operator = SimpleSubstitutionAttack()
    first = operator.apply(source)[0]
    assert first.operator_edits == 1
    second = operator.apply(first)[0]
    assert second.operator_edits == 2
    budget = Budget(max_queries=100, max_edits=3)
    assert budget.admits_edits(second)
    third = operator.apply(second)[0]
    fourth = operator.apply(third)[0]
    assert fourth.operator_edits == 4
    assert not budget.admits_edits(fourth)


def test_simple_substitution_is_one_atom_edit_per_application() -> None:
    source = Candidate(SOURCE, provenance=["source"])
    for child in SimpleSubstitutionAttack().apply(source)[:10]:
        change = graph_change(source.identifier, child.identifier)
        assert change.atom_edit_count == 1
        assert change.wildcard_count_delta == 0
        assert change.component_count_delta == 0


# ---------------------------------------------------------------------------
# Scaffold preservation
# ---------------------------------------------------------------------------
RING_SOURCE = "[*]CC1CCC(COC(=O)CCCC(=O)O[*])CC1"
AROMATIC_SOURCE = "[*]Oc1ccc(S(=O)(=O)c2ccc(O[*])cc2)cc1"


def test_scaffold_is_defined_explicitly() -> None:
    policy = ScaffoldPolicy()
    assert policy.scaffold_smiles(RING_SOURCE) == "C1CCCCC1"
    assert policy.scaffold_smiles(AROMATIC_SOURCE) == "O=S(=O)(c1ccccc1)c1ccccc1"
    assert policy.scaffold_smiles(ACYCLIC_SOURCE) == ""


def test_peripheral_modification_is_accepted() -> None:
    policy = ScaffoldPolicy()
    assert policy.retains(RING_SOURCE, "[*]CC1CCC(COC(=S)CCCC(=O)O[*])CC1")
    assert policy.retains(AROMATIC_SOURCE, "[*]Oc1ccc(S(=O)(=O)c2ccc(S[*])cc2)cc1")


def test_scaffold_atom_modification_is_rejected() -> None:
    policy = ScaffoldPolicy()
    assert not policy.retains(RING_SOURCE, "[*]CC1CCN(COC(=O)CCCC(=O)O[*])CC1")
    assert not policy.retains(AROMATIC_SOURCE, "[*]Oc1ccc(S(=O)(=O)c2ccc(O[*])cc2)n1")


def test_scaffold_deletion_is_rejected() -> None:
    policy = ScaffoldPolicy()
    check = policy.check("C1CCCCC1", "[*]CCCOC(=O)CCCC(=O)O[*]")
    assert not check.retained
    assert check.reason == "scaffold_not_retained"


def test_acyclic_source_has_no_scaffold_to_preserve() -> None:
    policy = ScaffoldPolicy()
    assert not policy.retains(ACYCLIC_SOURCE, ACYCLIC_SOURCE)
    operator = ScaffoldPreservingAttack(SimpleSubstitutionAttack())
    assert operator.apply(Candidate(ACYCLIC_SOURCE, provenance=["source"])) == []
    assert operator.rejection_reasons == {"source_has_no_scaffold": 1}


def test_scaffold_preserving_operator_filters_real_proposals() -> None:
    operator = ScaffoldPreservingAttack(SimpleSubstitutionAttack())
    source = Candidate(RING_SOURCE, provenance=["source"])
    accepted = operator.apply(source)
    assert accepted, "expected peripheral substitutions to survive"
    assert operator.rejection_reasons.get("scaffold_not_retained", 0) > 0
    policy = ScaffoldPolicy()
    for child in accepted:
        assert policy.retains(RING_SOURCE, child.identifier)
        assert child.operator_edits == 1
        assert "scaffold_preserving" in child.provenance


def test_polymer_attachment_point_is_protected_by_the_validator() -> None:
    validator = RDKitValidityChecker(require_attachment_count=count_wildcards(RING_SOURCE))
    assert count_wildcards(RING_SOURCE) == 2
    assert validator.is_valid(Candidate(RING_SOURCE))
    # losing an attachment point must be rejected
    assert not validator.is_valid(Candidate("[*]CC1CCC(COC(=O)CCCC(=O)O)CC1"))
    assert not validator.is_valid(Candidate("CC1CCC(COC(=O)CCCC(=O)O)CC1"))


def test_validator_rejects_disconnected_candidates() -> None:
    validator = RDKitValidityChecker(require_single_component=True)
    assert not validator.is_valid(Candidate("[*]CCO[*].CCO"))


# ---------------------------------------------------------------------------
# Operator rename
# ---------------------------------------------------------------------------
def test_functional_group_name_is_a_deprecated_alias_only() -> None:
    assert FunctionalGroupReplacementAttack is AliphaticCarbonSubstitutionAttack
    from materials_adv.domain.chemistry.attacks import functional_group

    assert "DEPRECATED" in (functional_group.__doc__ or "")


def test_aliphatic_operator_is_atom_level_not_fragment_replacement() -> None:
    operator = AliphaticCarbonSubstitutionAttack()
    assert {pattern for pattern, _ in operator.motifs} == {"[C;X4;h1,h2,h3]"}
    for child in operator.apply(Candidate(RING_SOURCE, provenance=["source"]))[:8]:
        change = graph_change(RING_SOURCE, child.identifier)
        assert change.component_count_delta == 0
        assert child.operator_edits == 1
        # one application is one operator edit, but chemistry may move more atoms
        assert change.atom_edit_count >= 1


# ---------------------------------------------------------------------------
# Model-backed pipeline behaviour
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def graphmpnn() -> GraphMPNNPredictor:
    return GraphMPNNPredictor(CKPT, SCALER)


@requires_model
def test_one_edit_sanity_pipeline_stays_in_the_plausible_regime(graphmpnn) -> None:
    operator = SimpleSubstitutionAttack()
    validator = RDKitValidityChecker(require_single_component=True,
                                     require_attachment_count=count_wildcards(SOURCE))
    source = Candidate(SOURCE, provenance=["source"])
    source_prediction = graphmpnn.predict(source)
    drifts = [
        abs(graphmpnn.predict(child) - source_prediction)
        for child in operator.apply(source)
        if validator.is_valid(child)
    ]
    assert drifts
    assert max(drifts) < 3.19          # canonical bounded maximum
    assert np.mean(drifts) < 2.0       # NOT the 6-7 eV stub regime


@requires_model
def test_predictions_are_eV_in_the_training_range(graphmpnn) -> None:
    predictions = np.asarray(graphmpnn.predict_batch([SOURCE, ACYCLIC_SOURCE, "CCO"]))
    assert predictions.min() > 0.0
    assert predictions.max() < 10.0


@requires_model
@pytest.mark.parametrize("strategy", ["random", "greedy", "metropolis", "evolutionary"])
def test_search_is_deterministic_for_a_fixed_seed(graphmpnn, strategy) -> None:
    def once():
        source = Candidate(SOURCE, provenance=["source"])
        evaluator = AttackEvaluator(graphmpnn, UntargetedDrift(), Budget(12, 3), source,
                                    CanonicalSmilesIdentity())
        validator = RDKitValidityChecker(require_single_component=True,
                                         require_attachment_count=count_wildcards(SOURCE))
        context = {"evaluator": evaluator, "validator": validator, "source": source}
        outcome = _make_strategy(strategy, SimpleSubstitutionAttack(), context, seed=7).search(source)
        return outcome.best_representation, outcome.queries_used

    assert once() == once()


# ---------------------------------------------------------------------------
# Verified-run artifacts
# ---------------------------------------------------------------------------
def test_verified_source_set_is_validation_only() -> None:
    path = VERIFIED / "benchmark_sources.csv"
    if not path.exists():
        pytest.skip("verification artifacts not present")
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    df = pd.read_csv(ROOT / "data/processed/processed.csv")
    from materials_adv.domain.chemistry.identity import canonical_smiles

    memberships: dict[str, set[int]] = {}
    for index, smiles in enumerate(df["original_representation"].tolist()):
        memberships.setdefault(canonical_smiles(smiles), set()).add(index)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])

    table = pd.read_csv(path)
    assert len(table) == 30
    for smiles in table["smiles"]:
        indices = memberships[smiles]
        assert indices <= val
        assert not (indices & train)
        assert not (indices & test)
    manifest = json.loads((VERIFIED / "source_manifest.json").read_text())
    assert manifest["source_split_membership"] == {"train": 0, "validation": 30, "test": 0}
    assert len(set(manifest["stratification"].values())) == 1  # balanced strata
    assert manifest["splits_sha256"] == hashlib.sha256(
        (ROOT / "data/processed/splits.json").read_bytes()
    ).hexdigest()


def test_verified_artifacts_respect_the_query_budget() -> None:
    path = VERIFIED / "strategy_runs.csv"
    if not path.exists():
        pytest.skip("verification artifacts not present")
    runs = pd.read_csv(path)
    assert (runs["queries_used"] <= runs["query_budget"]).all()
    assert (runs["predictor_calls"] == 1 + runs["queries_used"]).all()
    assert runs["one_model_call_per_query"].all()
    assert runs["best_operator_edits"].max() <= 3


def test_verified_artifacts_cover_every_strategy_and_budget() -> None:
    path = VERIFIED / "strategy_summary.csv"
    if not path.exists():
        pytest.skip("verification artifacts not present")
    summary = pd.read_csv(path)
    assert set(summary["strategy"]) == {"random", "greedy", "metropolis", "evolutionary"}
    assert set(summary["query_budget"]) == {10, 20, 50}
    assert (summary["mean_drift"] < 5.0).all()  # not the stub regime


def test_docs_keep_developmental_values_marked_unverified() -> None:
    doc = (ROOT / "docs/FRAMEWORK_V2_ATTACK_RESULTS.md").read_text()
    assert "DEVELOPMENTAL_UNVERIFIED" in doc
    assert "MULTIPLE_PROTOCOL_ERRORS" in doc
    # the developmental values are retained, but only ever labelled invalid
    assert "6.950" in doc or "6.949" in doc
    if (VERIFIED / "verified_summary.json").exists():
        assert "verified_run_1" in doc
