import pandas as pd
import json
import string
import os

def test_same_source_pool():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    pools = df.groupby(["attack_condition", "search_budget"])["source_id"].apply(set).tolist()
    if len(pools) > 1:
        for p in pools[1:]:
            assert p == pools[0]
            
def test_same_source_pool_materials():
    df = pd.read_csv("results/materials/per_example_results.csv")
    pools = df.groupby(["attack_condition", "search_budget"])["source_id"].apply(set).tolist()
    if len(pools) > 1:
        for p in pools[1:]:
            assert p == pools[0]

def test_manifest_filename_and_size():
    manifest = json.load(open("results/manifests/bio_cyber_sources_n30.json"))
    assert len(manifest) == 30
    m_manifest = json.load(open("results/manifests/materials_sources_n30.json"))
    assert len(m_manifest) == 30

def test_search_query_accounting():
    for f in ["results/bio_cyber/per_example_results.csv", "results/materials/per_example_results.csv"]:
        df = pd.read_csv(f)
        assert (df["search_queries"] >= 0).all()
        # Ensure budget was respected
        assert (df["search_queries"] <= df["search_budget"]).all()

def test_attribution_query_accounting():
    for f in ["results/bio_cyber/per_example_results.csv", "results/materials/per_example_results.csv"]:
        df = pd.read_csv(f)
        attr = df[df["attack_condition"].str.contains("Attribution")]
        assert (attr["attribution_queries"] > 0).all()
        non_attr = df[~df["attack_condition"].str.contains("Attribution")]
        assert (non_attr["attribution_queries"] == 0).all()

def test_total_query_arithmetic():
    for f in ["results/bio_cyber/per_example_results.csv", "results/materials/per_example_results.csv"]:
        df = pd.read_csv(f)
        assert (df["total_queries"] == df["search_queries"] + df["attribution_queries"]).all()

def test_deterministic_seeds():
    for f in ["results/bio_cyber/per_example_results.csv", "results/materials/per_example_results.csv"]:
        df = pd.read_csv(f)
        assert "seed" in df.columns
        assert (df["seed"] == 42).all()

def test_materials_rdkit_fields():
    df = pd.read_csv("results/materials/per_example_results.csv")
    assert "rdkit_valid" in df.columns
    assert "canonicalization_valid" in df.columns
    assert "tokenization_valid" in df.columns
    assert (df["rdkit_valid"] >= 0).all()

def test_materials_candidate_predictions_finite():
    df = pd.read_csv("results/materials/per_example_results.csv")
    assert df["candidate_prediction"].notna().all()

def test_materials_no_fabricated_success():
    df = pd.read_csv("results/materials/per_example_results.csv")
    assert df["success"].isna().all()
    
def test_cross_domain_table():
    df = pd.read_csv("results/cross_domain/attack_strategy_comparison.csv")
    assert "bio_cyber" in df["domain"].values
    assert "materials" in df["domain"].values

def test_materials_bootstrap_metadata():
    assert os.path.exists("results/materials/bootstrap_metadata.json")
    meta = json.load(open("results/materials/bootstrap_metadata.json"))
    assert meta["seed"] == 42
    assert meta["iterations"] >= 1000

