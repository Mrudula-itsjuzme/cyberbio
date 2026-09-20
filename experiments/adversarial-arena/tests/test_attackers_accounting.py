from arena.attackers import accounted_mcmc_adapter, evolutionary_adapter, random_smiles_adapter, mcmc_adapter
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
    metrics = a.get_metrics()
    assert metrics["target_queries"] >= 0
    assert metrics["query_budget_per_source"] == 5
    for r in records:
        assert r.target_query_count >= 0

def test_evo_accounting():
    a = evolutionary_adapter(seed=123, query_budget=5, max_edits=3)
    d = DummyDef()
    records = a.propose([SourceSample("S1", "C", 1.0)], d)
    assert isinstance(records, list)
    metrics = a.get_metrics()
    assert metrics["target_queries"] >= 0
    assert metrics["query_budget_per_source"] == 5
    for r in records:
        assert r.target_query_count >= 0

def test_randomsmiles_target_queries_zero():
    a = random_smiles_adapter(seed=123, proposal_budget=2)
    d = DummyDef()
    records = a.propose([SourceSample("S1", "C", 1.0)], d)
    assert len(a.source_accounting) == 1
    assert a.source_accounting[0]["attack_target_queries"] == 0

def test_existing_mcmc_preserves_reported_target_queries():
    a = mcmc_adapter(seed=123, allowed_tokens=["C", "O"], steps=5, proposal_budget=2)
    d = DummyDef()
    records = a.propose([SourceSample("S1", "C", 1.0)], d)
    assert len(a.source_accounting) == 1
    assert a.source_accounting[0]["attack_target_queries"] >= 0
