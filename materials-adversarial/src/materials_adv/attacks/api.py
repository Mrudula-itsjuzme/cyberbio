from typing import Any, List, Optional, Tuple
from materials_adv.framework.interfaces import (
    RepresentationAdapter, 
    ValidityChecker, 
    ConstraintSet, 
    AttackOperator, 
    Predictor, 
    Oracle, 
    SearchStrategy
)

def run_adversarial_attack(
    initial_obj: Any,
    predictor: Predictor,
    search_strategy: SearchStrategy,
    operators: List[AttackOperator],
    validity_checker: ValidityChecker,
    constraints: ConstraintSet,
    budget: int = 3,
    oracle: Optional[Oracle] = None
) -> Tuple[Any, float, Optional[float]]:
    """
    Core API for running an adversarial attack. 
    This is abstracted from the domain (e.g. chemistry, DNA).
    
    Args:
        initial_obj: The starting object (e.g. SMILES string or sequence).
        predictor: The surrogate model.
        search_strategy: The search algorithm (e.g., Metropolis, Greedy).
        operators: List of valid perturbation operators.
        validity_checker: Validates the domain object (e.g., RDKit Sanitization).
        constraints: Enforces similarity/budget constraints.
        budget: Maximum number of edits.
        oracle: The optional physical oracle (e.g., DFT). If None, no true error is computed.
        
    Returns:
        Tuple of (Best Candidate Object, Predictor Drift, True Oracle Error if oracle available).
    """
    
    best_candidate, max_drift = search_strategy.search(
        initial_obj, predictor, operators, validity_checker, constraints, budget
    )
    
    true_error = None
    if oracle is not None:
        # If oracle exists, compute E_S and E_G. E_S = |y_hat_adv - y_adv|
        predicted_val = predictor.predict([best_candidate])[0]
        true_val = oracle.evaluate(best_candidate)
        true_error = abs(predicted_val - true_val)
        
    return best_candidate, max_drift, true_error
