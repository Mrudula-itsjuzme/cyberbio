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

def test_manifest_filename_and_size():
    manifest = json.load(open("results/manifests/bio_cyber_sources_n30.json"))
    assert len(manifest) == 30

def test_no_fabricated_materials():
    assert not os.path.exists("results/materials/per_example_results.csv")

def test_source_hashes():
    manifest = json.load(open("results/manifests/bio_cyber_sources_n30.json"))
    for row in manifest:
        assert len(row["sequence_SHA256"]) == 64

def test_search_query_accounting():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    assert (df["search_queries"] > 0).all()
    # Random and MCMC don't stop early in this implementation
    rnd = df[df["attack_condition"].isin(["Random", "MCMC"])]
    assert (rnd["search_queries"] == rnd["search_budget"]).all()
    
    attr = df[df["attack_condition"].str.contains("Attribution")]
    assert (attr["search_queries"] == attr["search_budget"]).all()

def test_attribution_query_accounting():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    attr = df[df["attack_condition"].str.contains("Attribution")]
    # attribution_queries should equal the sequence length (which is usually around 150)
    assert (attr["attribution_queries"] > 0).all()
    
    non_attr = df[~df["attack_condition"].str.contains("Attribution")]
    assert (non_attr["attribution_queries"] == 0).all()

def test_total_query_arithmetic():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    assert (df["total_queries"] == df["search_queries"] + df["attribution_queries"]).all()

def test_runtime_includes_attribution():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    attr = df[df["attack_condition"].str.contains("Attribution")]
    assert (attr["attribution_runtime_seconds"] > 0).all()
    assert (attr["runtime_seconds"] >= attr["attribution_runtime_seconds"]).all()

def test_queries_to_success_semantics():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    succ = df[df["success"] == True]
    assert (succ["first_success_search_query"] <= succ["search_queries"]).all()
    assert (succ["first_success_total_query"] <= succ["total_queries"]).all()
    # If successful, first success search must be > 0
    assert (succ["first_success_search_query"] > 0).all()

def test_deterministic_seeds():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    assert "seed" in df.columns
    assert (df["seed"] == 42).all()

def test_aggregation_row_counts():
    agg = pd.read_csv("results/bio_cyber/aggregated_results.csv")
    assert len(agg) == 6 * 3 # 6 attacks * 3 budgets

def test_paired_comparison_alignment():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    df_50 = df[df["search_budget"] == 50]
    h = df_50[df_50["attack_condition"] == "Attribution-High"].sort_values("source_id")
    r = df_50[df_50["attack_condition"] == "Attribution-Random"].sort_values("source_id")
    assert (h["source_id"].values == r["source_id"].values).all()
