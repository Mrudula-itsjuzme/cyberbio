import pandas as pd
import json
import string

def test_attack_budget():
    df = pd.read_csv("results/v3/attack_results.csv")
    for idx, row in df.iterrows():
        assert row["number_of_edits"] <= row["budget"]
        
def test_alphabet_validity():
    df = pd.read_csv("results/v3/attack_results.csv")
    allowed = set("ACGT")
    for cand in df["candidate_sequence"].head(100):
        assert set(cand).issubset(allowed)
        
def test_same_source_pool():
    df = pd.read_csv("results/v3/attack_results.csv")
    pools = df.groupby(["attack", "budget"])["source_id"].apply(set).tolist()
    if len(pools) > 1:
        for p in pools[1:]:
            assert p == pools[0]

def test_frozen_bank_hashes():
    manifest = json.load(open("results/v3/frozen_attack_banks/manifest.json"))
    assert len(manifest) > 0
    for entry in manifest:
        assert "SHA256" in entry
        h = entry["SHA256"]
        assert len(h) == 64
        assert all(c in string.hexdigits for c in h)

def test_adaptive_target():
    df = pd.read_csv("results/v3/adaptive_attack_results.csv")
    assert df["defense"].iloc[0] == "cnn_adv_mixed"
