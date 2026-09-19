import pandas as pd
import json
import string

def test_same_source_pool():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    pools = df.groupby(["attack_family", "budget"])["source_id"].apply(set).tolist()
    if len(pools) > 1:
        for p in pools[1:]:
            assert p == pools[0]

def test_budget_adherence():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    for idx, row in df.iterrows():
        assert row["query_count"] <= row["budget"]

def test_edit_distance_correctness():
    df = pd.read_csv("results/bio_cyber/per_example_results.csv")
    for idx, row in df.iterrows():
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

def test_manifest_format():
    manifest = json.load(open("results/manifests/bio_cyber_sources.json"))
    assert len(manifest) > 0
    for entry in manifest:
        h = entry["SHA256"]
        assert len(h) == 64
        assert all(c in string.hexdigits for c in h)
