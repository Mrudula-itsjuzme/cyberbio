from arena.attackers import accounted_mcmc_adapter, evolutionary_adapter
from arena.contracts import SourceSample

class DummyDef:
    def predict(self, x): return [1.0 for _ in x]
    @property
    def calls(self): return 0
    @calls.setter
    def calls(self, v): pass

def test_mcmc_accounting():
    a = accounted_mcmc_adapter(seed=123, query_budget=5, max_edits=3)
    d = DummyDef()
    records = a.propose([SourceSample("S1", "C", 1.0)], d)
    assert isinstance(records, list)

def test_evo_accounting():
    a = evolutionary_adapter(seed=123, query_budget=5, max_edits=3)
    d = DummyDef()
    records = a.propose([SourceSample("S1", "C", 1.0)], d)
    assert isinstance(records, list)
