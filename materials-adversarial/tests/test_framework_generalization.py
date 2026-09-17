from materials_adv.framework.interfaces import SearchStrategy, Candidate, AttackOperator, Predictor
from materials_adv.framework.objectives import TargetIncrease
from materials_adv.framework.budget import Budget
from materials_adv.attacks.search.evolutionary import EvolutionarySearch

class StringValidator:
    def is_valid(self, candidate: Candidate) -> bool:
        # Only A, B, C allowed
        return all(c in "ABC" for c in candidate.identifier)

class StringSubstitutionAttack(AttackOperator):
    def apply(self, candidate: Candidate) -> list:
        candidates = []
        chars = list(candidate.identifier)
        for i in range(len(chars)):
            for rep in "ABC":
                if chars[i] != rep:
                    new_chars = chars.copy()
                    new_chars[i] = rep
                    c = Candidate(identifier="".join(new_chars), provenance=candidate.provenance + ["subst"])
                    candidates.append(c)
        return candidates

class MockStringPredictor(Predictor):
    def predict(self, candidate: Candidate) -> float:
        # Count 'A's
        return float(candidate.identifier.count('A'))

def test_framework_generalization_engineering():
    # Prove the framework works purely on strings, 0 RDKit imports needed
    source = Candidate("BBB", provenance=["source"])
    validator = StringValidator()
    assert validator.is_valid(source)
    
    operator = StringSubstitutionAttack()
    predictor = MockStringPredictor()
    objective = TargetIncrease()
    budget = Budget(max_queries=20, max_edits=3)
    
    search = EvolutionarySearch(
        operator=operator,
        objective=objective,
        predictor=predictor,
        budget=budget,
        population_size=2,
        elite_fraction=0.5,
        validator=validator
    )
    
    result = search.search(source)
    
    # We expect the search to maximize 'A's
    assert predictor.predict(result) > predictor.predict(source)
    assert budget.queries_used > 0
    assert budget.validate_edit_distance(result, source)

