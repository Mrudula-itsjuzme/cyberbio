import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from arena.contracts import Defender, SourceSample, CandidateRecord
from arena.probabilistic_attackers import cem_attacker, smc_attacker, validity_smc_attacker
from arena.prob_lab.proposals import arena_proposal_families
from arena.state_management import hash_attacker_state, snapshot_attacker_state, restore_attacker_state, evaluate_attacker_state_pure
from typing import Sequence, Any

class DummyDefender(Defender):
    round_id = 0
    checkpoint_id = "dummy"
    def predict(self, inputs: Sequence[str]) -> list[float]:
        return [0.5 for _ in inputs]
    def train(self, clean: Sequence[SourceSample], hard_examples: Sequence[CandidateRecord], *, round_id: int) -> dict[str, Any]:
        return {}
    def evaluate(self, sources: Sequence[SourceSample]) -> dict[str, Any]:
        return {"clean_rmse": 0.1, "mean_robustness_drift": 0.0}
    def checkpoint(self, path: str) -> str:
        return path
    def load_checkpoint(self, path: str) -> None:
        pass
    def get_metrics(self) -> dict[str, Any]:
        return {"clean_rmse": 0.1}

def get_dummy_samples():
    return [SourceSample("id1", "C", 1.0)]

# --- RESET TESTS ---

def test_cem_reset_state():
    attacker = cem_attacker(seed=42, query_budget=5, carry_state=False)
    initial_hash = hash_attacker_state(attacker)
    attacker._probs = [0.1, 0.9]
    attacker.reset(42)
    assert hash_attacker_state(attacker) == initial_hash, "CEM state not fully restored"

def test_smc_reset_state():
    attacker = smc_attacker(seed=42, query_budget=5, carry_state=False)
    initial_hash = hash_attacker_state(attacker)
    attacker._particle_state = ["CC"]
    attacker._weight_state = [1.0]
    attacker.reset(42)
    assert hash_attacker_state(attacker) == initial_hash, "SMC state not fully restored"

def test_validity_smc_reset_state():
    attacker = validity_smc_attacker(seed=42, query_budget=5, carry_state=False)
    initial_hash = hash_attacker_state(attacker)
    f = arena_proposal_families()[0].name
    attacker._validity_model.update(f, True)
    attacker.reset(42)
    assert hash_attacker_state(attacker) == initial_hash, "ValiditySMC state not fully restored"

# --- CARRY HASH CHAIN TESTS ---

def run_carry_chain(attacker):
    defender = DummyDefender()
    samples = get_dummy_samples()
    # R1
    attacker.propose(samples, defender)
    hash_after_t = hash_attacker_state(attacker)
    
    # We do not call reset() in carry mode
    hash_before_t1 = hash_attacker_state(attacker)
    assert hash_after_t == hash_before_t1, f"{attacker.name} carry hash chain broken"

def test_cem_carry_hash_chain():
    run_carry_chain(cem_attacker(seed=42, query_budget=10, carry_state=True))

def test_smc_carry_hash_chain():
    run_carry_chain(smc_attacker(seed=42, query_budget=10, carry_state=True))

def test_validity_smc_carry_hash_chain():
    run_carry_chain(validity_smc_attacker(seed=42, query_budget=10, carry_state=True))

# --- CROSS EVAL PURITY TESTS ---

def test_cem_cross_eval_pure():
    evaluate_attacker_state_pure(cem_attacker(seed=42, query_budget=10, carry_state=True), get_dummy_samples(), DummyDefender())

def test_smc_cross_eval_pure():
    evaluate_attacker_state_pure(smc_attacker(seed=42, query_budget=10, carry_state=True), get_dummy_samples(), DummyDefender())

def test_validity_smc_cross_eval_pure():
    evaluate_attacker_state_pure(validity_smc_attacker(seed=42, query_budget=10, carry_state=True), get_dummy_samples(), DummyDefender())

