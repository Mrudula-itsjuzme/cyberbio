import pytest
from materials_adv.framework.interfaces import (
    RepresentationAdapter,
    ValidityChecker,
    ConstraintSet,
    AttackOperator,
    Predictor,
    Oracle,
    SearchStrategy
)
from materials_adv.attacks.api import run_adversarial_attack

class MockPredictor(Predictor):
    def predict(self, batch):
        return [0.0 for _ in batch]

class MockOperator(AttackOperator):
    def apply(self, obj):
        return [obj + "_mod"]

class MockSearch(SearchStrategy):
    def search(self, initial_obj, predictor, operators, validity_checker, constraints, budget):
        return initial_obj + "_mod", 1.0

def test_framework_interfaces():
    assert issubclass(MockPredictor, Predictor)
    assert issubclass(MockOperator, AttackOperator)
    assert issubclass(MockSearch, SearchStrategy)

def test_run_adversarial_attack_no_oracle():
    predictor = MockPredictor()
    search = MockSearch()
    operator = MockOperator()
    
    best_candidate, max_drift, true_error = run_adversarial_attack(
        initial_obj="CC",
        predictor=predictor,
        search_strategy=search,
        operators=[operator],
        validity_checker=None,
        constraints=None,
        budget=1
    )
    
    assert best_candidate == "CC_mod"
    assert max_drift == 1.0
    assert true_error is None
