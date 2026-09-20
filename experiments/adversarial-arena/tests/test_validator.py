from arena.probabilistic_attackers import _ArenaValidator

class MockCanonical:
    def validate_pair(self, source, candidate):
        raise ValueError("simulated validator failure")

def test_validator_exception_never_becomes_eligible():
    val = _ArenaValidator()
    val._v = MockCanonical()
    res = val.validate("C", "CC")
    assert res.eligible == False
    assert res.representation_valid == False
    assert res.domain_constraint_pass == False
    assert res.reasons[0] == "validator_exception:ValueError"
