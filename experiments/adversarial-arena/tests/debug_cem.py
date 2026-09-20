import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "materials-adversarial", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_state_invariants import cem_attacker, DummyDefender, get_dummy_samples
import copy
import json
attacker = cem_attacker(seed=42, query_budget=10, carry_state=True)
samples, defender = get_dummy_samples(), DummyDefender()
attacker.propose(samples, defender)
s1 = json.dumps(attacker.get_state())
temp_att = copy.deepcopy(attacker)
temp_att.propose(samples, defender)
s2 = json.dumps(attacker.get_state())
print("Same after deepcopy?", s1 == s2)
