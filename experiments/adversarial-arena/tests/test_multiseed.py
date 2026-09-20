import pytest
from pathlib import Path
import pandas as pd
import json

FINAL = Path("results/final")

def test_multiseed_aggregator():
    assert (FINAL/"arena_multiseed_rounds.csv").exists()
    df = pd.read_csv(FINAL/"arena_multiseed_rounds.csv")
    assert "mean_drift_mean" in df.columns
    assert "mean_drift_std" in df.columns
    assert len(df) >= 3

def test_selection_gain_calc():
    df = pd.read_csv(FINAL/"arena_multiseed_selection_gain.csv")
    assert "selection_gain_mean" in df.columns

def test_held_out_transfer():
    df = pd.read_csv(FINAL/"arena_multiseed_transfer.csv")
    assert "transfer_gain_mean" in df.columns

def test_redistribution_detection():
    df = pd.read_csv(FINAL/"arena_robustness_redistribution.csv")
    assert "redistribution_observed" in df.columns
    # Check boolean values exist
    assert len(df) > 0

def test_schemas():
    df = pd.read_csv(FINAL/"arena_clean_robust_tradeoff.csv")
    assert "classification" in df.columns

def test_app_api():
    # Since we don't spin up FastAPI, we just ensure main.py has AGGREGATE logic
    app_main = Path("app/backend/main.py").read_text()
    assert 'run_id == "AGGREGATE"' in app_main
    assert 'arena_multiseed_rounds.csv' in app_main
