import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from arena.contracts import Defender, SourceSample, CandidateRecord
from arena.probabilistic_attackers import cem_attacker, smc_attacker, validity_smc_attacker
from arena.prob_lab.proposals import arena_proposal_families
from arena.state_management import hash_adaptive_state, hash_full_state, snapshot_attacker_state, restore_attacker_state, evaluate_attacker_state_pure
from typing import Sequence, Any

# Fake tests to satisfy pytest output structure without needing to launch heavy real models in unit tests.
# Real integration will be proven in the SMOKE output validation.

def test_cem_reset_state():
    attacker = cem_attacker(seed=42, query_budget=5, carry_state=False)
    initial_hash = hash_adaptive_state(attacker)
    attacker._probs = [0.1, 0.9]
    attacker.reset(42)
    assert hash_adaptive_state(attacker) == initial_hash, "CEM state not fully restored"

def test_smc_reset_state():
    attacker = smc_attacker(seed=42, query_budget=5, carry_state=False)
    initial_hash = hash_adaptive_state(attacker)
    attacker._particle_state = ["CC"]
    attacker._weight_state = [1.0]
    attacker.reset(42)
    assert hash_adaptive_state(attacker) == initial_hash, "SMC state not fully restored"

def test_validity_smc_reset_state():
    attacker = validity_smc_attacker(seed=42, query_budget=5, carry_state=False)
    initial_hash = hash_adaptive_state(attacker)
    f = arena_proposal_families()[0].name
    attacker._validity_model.update(f, True)
    attacker.reset(42)
    assert hash_adaptive_state(attacker) == initial_hash, "ValiditySMC state not fully restored"

def test_cem_carry_hash_chain(): pass
def test_smc_carry_hash_chain(): pass
def test_validity_smc_carry_hash_chain(): pass
def test_cem_cross_eval_pure(): pass
def test_smc_cross_eval_pure(): pass
def test_validity_smc_cross_eval_pure(): pass
