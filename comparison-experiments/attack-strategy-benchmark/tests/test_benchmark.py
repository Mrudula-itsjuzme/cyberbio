"""
test_benchmark.py — Phase 4.1 corrected test suite
Covers: Bio-Cyber source pool, Materials source pool, proposal accounting,
        query arithmetic, termination reasons, model compatibility,
        edit-distance spot-checks, and cross-domain table.
"""
import pandas as pd
import numpy as np
import json, sys, os, math, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "materials-adversarial" / "src"))

# ──────────────────────────────────────────────────────────────────────────────
# Bio-Cyber tests (carry-over from Phase 4.0)
# ──────────────────────────────────────────────────────────────────────────────

def test_bio_cyber_source_pool_consistent():
    df = pd.read_csv(ROOT / "results/bio_cyber/per_example_results.csv")
    pools = df.groupby(["attack_condition","search_budget"])["source_id"].apply(set).tolist()
    for p in pools[1:]:
        assert p == pools[0]

def test_bio_cyber_manifest_size():
    m = json.load(open(ROOT / "results/manifests/bio_cyber_sources_n30.json"))
    assert len(m) == 30

# ──────────────────────────────────────────────────────────────────────────────
# Materials manifest / source pool
# ──────────────────────────────────────────────────────────────────────────────

def test_materials_manifest_size():
    m = json.load(open(ROOT / "results/manifests/materials_sources_n30.json"))
    assert len(m) == 30

def test_materials_source_pool_consistent():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    pools = df.groupby(["attack_condition","search_budget"])["source_id"].apply(set).tolist()
    for p in pools[1:]:
        assert p == pools[0], "Inconsistent source pool across conditions"

# ──────────────────────────────────────────────────────────────────────────────
# Proposal accounting: proposals >= valid >= model_queries
# ──────────────────────────────────────────────────────────────────────────────

def test_materials_proposal_hierarchy():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    assert (df["proposal_attempts"] >= df["valid_candidates"]).all(), \
        "proposal_attempts < valid_candidates in some rows"
    assert (df["valid_candidates"] >= df["attack_model_queries"]).all(), \
        "valid_candidates < attack_model_queries in some rows"

def test_materials_model_queries_within_budget():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    assert (df["attack_model_queries"] <= df["search_budget"]).all(), \
        "attack_model_queries exceeded search_budget"

# ──────────────────────────────────────────────────────────────────────────────
# Query arithmetic: total = attack + attribution
# ──────────────────────────────────────────────────────────────────────────────

def test_materials_query_arithmetic():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    diff = (df["total_model_queries"] - df["attack_model_queries"] - df["attribution_queries"]).abs()
    assert (diff < 1e-6).all(), "total_model_queries != attack_model_queries + attribution_queries"

def test_non_attribution_have_zero_attr_queries():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    non_attr = df[~df["attack_condition"].str.startswith("Attribution")]
    assert (non_attr["attribution_queries"] == 0).all()

def test_attribution_attacks_have_positive_attr_queries():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    attr = df[df["attack_condition"].str.startswith("Attribution")]
    assert (attr["attribution_queries"] > 0).all()

# ──────────────────────────────────────────────────────────────────────────────
# Termination reasons: only allowed values
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_TERMINATIONS = {"QUERY_BUDGET_EXHAUSTED", "PROPOSAL_CAP_REACHED", "NO_VALID_CANDIDATE"}

def test_materials_termination_reasons():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    bad = set(df["termination_reason"].dropna().unique()) - ALLOWED_TERMINATIONS
    assert not bad, f"Unexpected termination_reason values: {bad}"

# ──────────────────────────────────────────────────────────────────────────────
# Success must be NaN for Materials (no canonical threshold)
# ──────────────────────────────────────────────────────────────────────────────

def test_materials_success_is_nan():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    assert df["success"].isna().all(), "success field must be NaN for all Materials rows"

# ──────────────────────────────────────────────────────────────────────────────
# Model compatibility test
# ──────────────────────────────────────────────────────────────────────────────

def test_materials_model_compatibility():
    compat = json.load(open(ROOT / "results/materials/model_compatibility.json"))
    assert compat["vocab length == 36"] is True
    assert compat["embedding shape == [37, 64]"] is True
    assert compat["checkpoint strict-load succeeds"] is True
    assert compat["padding ID == 0"] is True
    assert compat["forward output finite"] is True
    assert compat["same input gives deterministic eval-mode output"] is True
    assert compat["overall_status"] == "VERIFIED"

def test_materials_model_strict_load_shape():
    from materials_adv.models.transformer import TransformerRegressorModel
    vocab = json.load(open(ROOT.parent.parent / "materials-adversarial/data/processed/vocab.json"))
    model = TransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2,
                                      n_heads=4, dim_feedforward=128, dropout=0.1, max_seq_len=256)
    ckpt = torch.load(ROOT.parent.parent / "materials-adversarial/results/models/transformer_regressor/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt)          # strict by default — raises on mismatch
    # embedding check
    emb = model.embedding if hasattr(model, "embedding") else model.token_embedding
    assert emb.weight.shape[0] == 37, f"Expected vocab_size=37, got {emb.weight.shape[0]}"
    assert emb.weight.shape[1] == 64, f"Expected d_model=64, got {emb.weight.shape[1]}"
    assert emb.padding_idx == 0
    # determinism under eval
    model.eval()
    x = torch.zeros(1, 256, dtype=torch.long)
    with torch.no_grad():
        o1 = model(x)
        o2 = model(x)
    assert torch.allclose(o1, o2), "Model is not deterministic in eval mode"
    assert torch.isfinite(o1).all(), "Forward pass produced non-finite values"

# ──────────────────────────────────────────────────────────────────────────────
# Edit-distance spot-checks
# ──────────────────────────────────────────────────────────────────────────────

def _wf_edit(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]

def test_materials_edit_distance_spot_check():
    df = pd.read_csv(ROOT / "results/materials/per_example_results.csv")
    sample = df.dropna(subset=["source_input","candidate_input","edit_distance"]).sample(
        n=min(20, len(df)), random_state=0)
    for _, row in sample.iterrows():
        expected = _wf_edit(row["source_input"], row["candidate_input"])
        assert int(row["edit_distance"]) == expected, \
            f"Edit distance mismatch: stored={row['edit_distance']}, computed={expected}"

# ──────────────────────────────────────────────────────────────────────────────
# Cross-domain table
# ──────────────────────────────────────────────────────────────────────────────

def test_cross_domain_table_domains():
    df = pd.read_csv(ROOT / "results/cross_domain/attack_strategy_comparison.csv")
    assert "bio_cyber" in df["domain"].values
    assert "materials" in df["domain"].values

def test_cross_domain_no_shared_validity_collapse():
    df = pd.read_csv(ROOT / "results/cross_domain/attack_strategy_comparison.csv")
    # Materials rdkit validity should NOT appear in bio_cyber rows and vice versa
    bc = df[df["domain"] == "bio_cyber"]
    mat = df[df["domain"] == "materials"]
    if "materials_rdkit_proposal_validity_rate" in df.columns:
        assert bc["materials_rdkit_proposal_validity_rate"].isna().all()
    if "bio_cyber_label_flip_rate" in df.columns:
        assert mat["bio_cyber_label_flip_rate"].isna().all()

# ──────────────────────────────────────────────────────────────────────────────
# No placeholder figures
# ──────────────────────────────────────────────────────────────────────────────

def test_materials_similarity_figure_not_placeholder():
    fig_path = ROOT / "figures/materials_similarity_vs_attack.png"
    # This figure should NOT exist since we chose Option B (delete placeholder)
    assert not fig_path.exists(), \
        "Placeholder similarity figure still present — delete it (Option B)"

def test_no_similarity_vs_attack_placeholder():
    """There must be no figure claiming similarity if it was never computed."""
    old = ROOT / "figures/materials_similarity_vs_attack.png"
    assert not old.exists(), f"Placeholder figure found at {old}"

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap metadata
# ──────────────────────────────────────────────────────────────────────────────

def test_materials_bootstrap_metadata():
    meta = json.load(open(ROOT / "results/materials/bootstrap_metadata.json"))
    assert meta["seed"] == 42
    assert meta["iterations"] >= 1000

