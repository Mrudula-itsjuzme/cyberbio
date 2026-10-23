import pytest
import sys
import argparse
from pathlib import Path

comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from scripts.run_comparison import parse_args

def test_parse_args_basic():
    args = parse_args(["--attack", "random", "--source-count", "10", "--query-budget", "15"])
    assert "random" in args.attack
    assert args.source_count == 10
    assert args.query_budget == 15
    assert args.seed == 42 # Default

def test_parse_args_multiple_attacks():
    args = parse_args(["--attack", "random", "mcmc", "--seed", "123"])
    assert "random" in args.attack
    assert "mcmc" in args.attack
    assert args.seed == 123
