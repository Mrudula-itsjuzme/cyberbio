import pytest
from materials_adv.framework.interfaces import (
    SearchStrategy, AttackOperator, Predictor, ConstraintSet, Candidate,
    RepresentationAdapter, ValidityChecker
)
from materials_adv.domain.chemistry.rdkit_adapter import SMILESAdapter, RDKitValidityChecker
from materials_adv.attacks.api import run_adversarial_attack

class DummyPredictor(Predictor):
    def predict(self, batch):
        # Dummy prediction based on string length
        return [float(len(str(b))) for b in batch]

class DummySubstitution(AttackOperator):
    def apply(self, obj):
        # Simply append a Carbon
        try:
            from materials_adv.framework.interfaces import Candidate
            smiles = str(obj.identifier)
            return [Candidate(identifier=smiles + "C")]
        except Exception:
            return []

class DummyValidityChecker(ValidityChecker):
    def is_valid(self, candidate: Candidate) -> bool:
        return True

class DummyConstraint(ConstraintSet):
    def check_constraints(self, original_obj, candidate_obj):
        return True

class GreedySearch(SearchStrategy):
    def search(self, initial_obj, predictor, operators, validity_checker, constraints, budget):
        current_obj = initial_obj
        for _ in range(budget):
            candidates = []
            for op in operators:
                candidates.extend(op.apply(current_obj))
            
            valid_candidates = [c for c in candidates if validity_checker.is_valid(c) and constraints.check_constraints(initial_obj, c)]
            if not valid_candidates:
                break
                
            scores = predictor.predict(valid_candidates)
            best_idx = scores.index(max(scores))
            current_obj = valid_candidates[best_idx]
            
        return current_obj, max(predictor.predict([current_obj]))

def test_end_to_end_abstraction():
    adapter = SMILESAdapter()
    validity = DummyValidityChecker()
    operator = DummySubstitution()
    constraints = DummyConstraint()
    predictor = DummyPredictor()
    search = GreedySearch()
    
    initial_mol = adapter.decode("CC")
    
    best_candidate, max_drift, true_error = run_adversarial_attack(
        initial_obj=initial_mol,
        predictor=predictor,
        search_strategy=search,
        operators=[operator],
        validity_checker=validity,
        constraints=constraints,
        budget=2
    )
    
    best_smiles = adapter.encode(best_candidate)
    assert best_smiles == "CCCC"
