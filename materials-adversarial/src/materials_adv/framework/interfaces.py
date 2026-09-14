from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

class RepresentationAdapter(ABC):
    """
    Handles conversion between sequence (string) and native object representations.
    """
    @abstractmethod
    def to_object(self, representation: Any) -> Any:
        pass
        
    @abstractmethod
    def to_representation(self, obj: Any) -> Any:
        pass

class ValidityChecker(ABC):
    """
    Checks if a given domain object is structurally valid.
    """
    @abstractmethod
    def is_valid(self, obj: Any) -> bool:
        pass

class ConstraintSet(ABC):
    """
    Checks if an object satisfies a set of domain-specific constraints.
    """
    @abstractmethod
    def check_constraints(self, original_obj: Any, candidate_obj: Any) -> bool:
        pass

class AttackOperator(ABC):
    """
    An operator that proposes perturbations to a domain object.
    """
    @abstractmethod
    def apply(self, obj: Any) -> List[Any]:
        pass

class Predictor(ABC):
    """
    The surrogate machine learning model.
    """
    @abstractmethod
    def predict(self, batch: List[Any]) -> List[float]:
        pass

class Oracle(ABC):
    """
    The physical ground-truth oracle.
    """
    @abstractmethod
    def evaluate(self, obj: Any) -> float:
        pass

class SearchStrategy(ABC):
    """
    A search algorithm to find adversarial examples.
    """
    @abstractmethod
    def search(self, 
               initial_obj: Any, 
               predictor: Predictor, 
               operators: List[AttackOperator],
               validity: ValidityChecker,
               constraints: ConstraintSet,
               budget: int) -> Tuple[Any, float]:
        pass
