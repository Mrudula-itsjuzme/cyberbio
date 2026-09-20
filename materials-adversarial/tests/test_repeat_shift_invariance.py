import json
from pathlib import Path

import pandas as pd
import pytest

from materials_adv.robustness.repeat_shift import (
    generate_repeat_shift,
    is_repeat_shift_equivalent,
    periodic_signature,
    repeat_variant,
    shift_variants,
    tokenizer_roundtrip,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "repeat_shift_invariance"


@pytest.mark.parametrize(
    "source",
    ["[*]CCO[*]", "[*]COC(C)C[*]", "[*]c1ccc(CC[*])cc1"],
)
def test_shift_preserves_periodic_graph(source):
    variants = shift_variants(source)
    assert variants
    assert all(periodic_signature(v.variant) == periodic_signature(source) for v in variants)


def test_repeat_is_exact_tiling():
    variant = repeat_variant("[*]CCO[*]", 2)
    assert variant is not None
    assert variant.equivalence_status == "VERIFIED_EQUIVALENT"
    ok, reason, metadata = is_repeat_shift_equivalent(
        variant.source, variant.variant, "REPEAT", {"repeat_factor": 2}
    )
    assert ok and reason == "exact_tiled_repeat_graph_match"
    assert metadata["expected_repeat"] == variant.variant


def test_attachment_mismatch_is_rejected():
    ok, reason, _ = is_repeat_shift_equivalent(
        "[*]CCO[*]", "[*]CCO", "SHIFT", {"repeat_factor": 1}
    )
    assert not ok
    assert reason == "invalid_or_non_linear_attachment_semantics"


def test_chemistry_changing_shift_is_rejected():
    ok, reason, _ = is_repeat_shift_equivalent(
        "[*]CCO[*]", "[*]CCN[*]", "SHIFT", {"repeat_factor": 1}
    )
    assert not ok and reason == "periodic_graph_mismatch"


def test_orientation_sensitive_stereochemistry_change_is_rejected():
    ok, reason, _ = is_repeat_shift_equivalent(
        "[*]C[C@H](F)CO[*]",
        "[*]C[C@@H](F)CO[*]",
        "SHIFT",
        {"repeat_factor": 1},
    )
    assert not ok and reason == "stereochemistry_mismatch"


def test_invalid_repeat_is_rejected():
    ok, reason, _ = is_repeat_shift_equivalent(
        "[*]CCO[*]", "[*]CCCCCC[*]", "REPEAT", {"repeat_factor": 2}
    )
    assert not ok and reason == "repeat_graph_mismatch"


def test_generation_is_deterministic_and_unique():
    first = generate_repeat_shift("[*]CCOC(C)C[*]")
    second = generate_repeat_shift("[*]CCOC(C)C[*]")
    assert [v.record() for v in first] == [v.record() for v in second]
    assert len({v.variant for v in first}) == len(first)


def test_unknown_and_max_length_accounting():
    vocab = json.loads((ROOT / "data/processed/vocab.json").read_text())
    clean = tokenizer_roundtrip("[*]CCO[*]", vocab)
    unknown = tokenizer_roundtrip("[*][Xe]CC[*]", vocab)
    too_long = tokenizer_roundtrip("[*]" + "C" * 255 + "[*]", vocab)
    assert clean["roundtrip"] and clean["unknown_count"] == 0
    assert unknown["unknown_count"] == 1
    assert not too_long["within_max_length"]


def test_same_phase5b_cohort_is_reused_without_substitution():
    original = json.loads(
        (ROOT / "results/representation_equivalence_sequence/cohort_manifest.json").read_text()
    )["sources"]
    applicability = pd.read_csv(RESULTS / "cohort_applicability.csv")
    assert applicability.source_id.tolist() == [row["source_id"] for row in original]
    assert len(applicability) == 100


def test_matched_k_output_obeys_availability_and_pairing():
    matched = pd.read_csv(RESULTS / "matched_k_comparison.csv")
    variants = pd.read_csv(RESULTS / "variants.csv")
    broader = pd.read_csv(
        ROOT / "results/representation_equivalence_validation/token_pair_audit.csv"
    )
    assert set(matched.k).issubset({1, 2, 5, 10, 20})
    for row in matched.itertuples():
        assert (variants.source_id == row.source_id).sum() >= row.k
        assert (broader.source_id == row.source_id).sum() >= row.k
        assert row.paired_difference == pytest.approx(
            row.broader_expected_worst - row.repeat_shift_expected_worst,
            abs=1e-6,
        )


def test_graph_control_is_matched_pair_for_pair():
    variants = pd.read_csv(RESULTS / "variants.csv")
    control = pd.read_csv(RESULTS / "graph_control.csv")
    assert list(zip(control.source_id, control.variant_id, control.family)) == list(
        zip(variants.source_id, variants.variant_id, variants.family)
    )
    assert control.graph_absolute_drift.notna().all()
