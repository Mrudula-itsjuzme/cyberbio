"""Legacy evolutionary search.

DEPRECATED for new work: use :class:`materials_adv.framework.search.EvolutionarySearch`,
which scores exclusively through
:class:`~materials_adv.framework.accounting.AttackEvaluator` and therefore cannot spend
queries off-budget. This class is preserved verbatim because the frozen forensic audit
(``scripts/audit_framework_v2_benchmark.py``) reproduces its exact behaviour.

Import hygiene: this module no longer imports RDKit at import time. The chemistry
validator default is resolved lazily inside ``__init__``, so importing the generic search
layer does not drag RDKit into ``sys.modules`` (audit section 19).
"""

import random
from typing import List

from materials_adv.framework.interfaces import SearchStrategy, Candidate, AttackOperator, Predictor
from materials_adv.framework.objectives import AttackObjective
from materials_adv.framework.budget import Budget

class EvolutionarySearch(SearchStrategy):
    """
    Evolutionary search algorithm targeting property drift.

    Query accounting: this class asks the objective to score, and the objective asks
    the predictor twice per nominal query. It is kept for archival reproducibility
    only; it is NOT budget-correct. Use
    :class:`materials_adv.framework.search.EvolutionarySearch` for measured work.
    """
    def __init__(self, 
                 operator: AttackOperator, 
                 objective: AttackObjective, 
                 predictor: Predictor, 
                 budget: Budget,
                 population_size: int = 10,
                 elite_fraction: float = 0.2,
                 validator = None):
        self.operator = operator
        self.objective = objective
        self.predictor = predictor
        self.budget = budget
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        if validator is None:
            from materials_adv.domain.chemistry.validator import RDKitValidityChecker
            self.validator = RDKitValidityChecker()
        else:
            self.validator = validator

    def search(self, source: Candidate) -> Candidate:
        # Deterministic random state should be set outside, but keeping it clean here.
        population = [source]
        best_candidate = source
        best_score = float('-inf')
        
        while not self.budget.is_exhausted():
            # Generate new candidates
            new_candidates = []
            for p in population:
                mutated = self.operator.apply(p)
                for m in mutated:
                    # STRICT EDIT BUDGET ENFORCEMENT
                    if self.validator.is_valid(m) and self.budget.validate_edit_distance(m, source):
                        new_candidates.append(m)
            
            if not new_candidates:
                break
                
            # Score and deduplicate
            scored = []
            seen = set()
            for c in new_candidates:
                if c.identifier in seen:
                    continue
                seen.add(c.identifier)
                
                if not self.budget.use_query():
                    break
                score = self.objective.evaluate(self.predictor, source, c)
                scored.append((score, c))
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
            
            if not scored:
                break
                
            # Select elites
            scored.sort(key=lambda x: x[0], reverse=True)
            elite_count = max(1, int(self.population_size * self.elite_fraction))
            population = [x[1] for x in scored[:elite_count]]
            
            # Repopulate (simple mutation from elites)
            # Make sure we don't exceed budget or population size
            repopulated = []
            attempts = 0
            while len(population) + len(repopulated) < self.population_size and not self.budget.is_exhausted() and attempts < 100:
                attempts += 1
                parent = random.choice(population[:elite_count])
                mutated = self.operator.apply(parent)
                if mutated:
                    m = random.choice(mutated)
                    if self.validator.is_valid(m) and self.budget.validate_edit_distance(m, source):
                        repopulated.append(m)
                        attempts = 0
            population.extend(repopulated)
                    
        return best_candidate
