import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arena.probabilistic_attackers import cem_attacker, smc_attacker, validity_smc_attacker
from arena.prob_lab.proposals import arena_proposal_families

def test_cem_reset():
    attacker = cem_attacker(seed=42, query_budget=5, carry_state=False)
    attacker._probs = [0.1, 0.9]
    attacker.reset(42)
    assert attacker._probs is None, "CEM probs not reset"

def test_smc_reset():
    attacker = smc_attacker(seed=42, query_budget=5, carry_state=False)
    attacker._particle_state = ["CC"]
    attacker.reset(42)
    assert attacker._particle_state is None, "SMC particle state not reset"

def test_validity_smc_reset():
    attacker = validity_smc_attacker(seed=42, query_budget=5, carry_state=False)
    f = arena_proposal_families()[0].name
    attacker._validity_model.update(f, True)
    assert attacker._validity_model.snapshot()[f]["alpha"] == 2.0
    attacker.reset(42)
    assert attacker._validity_model.snapshot().get(f, {"alpha": 1.0})["alpha"] == 1.0, "ValiditySMC validity model not reset"
    
def test_cem_carry():
    attacker = cem_attacker(seed=42, query_budget=5, carry_state=True)
    attacker._probs = [0.1, 0.9]
    # In carry_state, reset is not called between rounds by orchestration,
    # but let's verify setting the state works.
    pass

if __name__ == "__main__":
    test_cem_reset()
    test_smc_reset()
    test_validity_smc_reset()
    print("ALL TESTS PASSED")
