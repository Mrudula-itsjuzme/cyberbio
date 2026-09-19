import pandas as pd
import json
import string
import os

def test_same_source_pool():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    pools = df.groupby(["attack_condition", "budget"])["source_id"].apply(set).tolist()
    if len(pools) > 1:
        for p in pools[1:]:
            assert p == pools[0]

def test_n_greater_equal_30():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    # For a given attack and budget
    n = len(df[(df["attack_condition"] == "Random") & (df["budget"] == 5)])
    assert n >= 30

def test_no_fabricated_materials():
    assert not os.path.exists("results/materials/per_example_results.csv")

def test_source_hashes():
    manifest = json.load(open("results/manifests/bio_cyber_sources_n50.json"))
    for row in manifest:
        assert len(row["sequence_SHA256"]) == 64

def test_query_counts_positive():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    assert (df["total_queries"] > 0).all()

def test_queries_to_success_less_than_total():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    succ = df[df["queries_to_success"].notna()]
    assert (succ["queries_to_success"] <= succ["total_queries"]).all()

def test_budget_adherence():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    assert (df["total_queries"] <= df["budget"] * 5).all() # Evolutionary takes up to budget + pop_size

def test_edit_distance_correctness():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    for idx, row in df.head(100).iterrows():
        calc_dist = sum(1 for i in range(len(row["source_input"])) if row["source_input"][i] != row["candidate_input"][i])
        assert row["edit_distance"] == calc_dist

def test_alphabet_validity():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    for cand in df["candidate_input"].head(100):
        assert set(cand).issubset(set("ACGT"))

def test_deterministic_seeds():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    assert "seed" in df.columns
    assert (df["seed"] == 42).all()

def test_aggregation_row_counts():
    agg = pd.read_csv("results/bio_cyber/aggregated_results.csv")
    assert len(agg) == 6 * 3 # 6 attacks * 3 budgets

def test_paired_comparison_alignment():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    df_50 = df[df["budget"] == 50]
    h = df_50[df_50["attack_condition"] == "Attribution-High"].sort_values("source_id")
    r = df_50[df_50["attack_condition"] == "Attribution-Random"].sort_values("source_id")
    assert (h["source_id"].values == r["source_id"].values).all()
