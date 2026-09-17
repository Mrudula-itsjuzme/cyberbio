from materials_adv.framework.budget import Budget
from materials_adv.framework.objectives import TargetIncrease
from materials_adv.framework.interfaces import Candidate, Predictor

class MockPredictor(Predictor):
    def predict(self, representation):
        return representation.val

class MockCandidate(Candidate):
    def __init__(self, val):
        super().__init__(identifier=str(val))
        self.val = val

    def is_equivalent_to(self, other):
        return self.val == other.val

def test_budget():
    b = Budget(max_edits=3, max_queries=2)
    assert b.use_query()
    assert b.use_query()
    assert not b.use_query()
    assert b.is_exhausted()

def test_objectives():
    pred = MockPredictor()
    obj = TargetIncrease()
    c_src = MockCandidate(1.0)
    c_adv = MockCandidate(3.5)
    assert obj.evaluate(pred, c_src, c_adv) == 2.5
