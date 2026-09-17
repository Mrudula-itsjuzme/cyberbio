with open("src/materials_adv/framework/budget.py", "w") as f:
    f.write("""class Budget:
    def __init__(self, max_queries: int, max_edits: int):
        self.max_queries = max_queries
        self.max_edits = max_edits
        self.queries_used = 0

    def use_query(self) -> bool:
        if self.queries_used >= self.max_queries:
            return False
        self.queries_used += 1
        return True

    def is_exhausted(self) -> bool:
        return self.queries_used >= self.max_queries
        
    def validate_edit_distance(self, candidate, source) -> bool:
        # In a real implementation this would calculate levenshtein or graph distance
        # For our framework, we assume candidate stores edit history in provenance
        # This prevents Phase 12-style edit creep.
        edit_count = sum(1 for p in candidate.provenance if p != "source")
        return edit_count <= self.max_edits
""")

with open("src/materials_adv/attacks/search/evolutionary.py", "w") as f:
    f.write("""import random
from typing import List

from materials_adv.framework.interfaces import SearchStrategy, Candidate, AttackOperator, Predictor
from materials_adv.framework.objectives import AttackObjective
from materials_adv.framework.budget import Budget
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

class EvolutionarySearch(SearchStrategy):
    \"\"\"
    Evolutionary search algorithm targeting property drift.
    \"\"\"
    def __init__(self, 
                 operator: AttackOperator, 
                 objective: AttackObjective, 
                 predictor: Predictor, 
                 budget: Budget,
                 population_size: int = 10,
                 elite_fraction: float = 0.2):
        self.operator = operator
        self.objective = objective
        self.predictor = predictor
        self.budget = budget
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.validator = RDKitValidityChecker()

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
            while len(population) + len(repopulated) < self.population_size and not self.budget.is_exhausted():
                parent = random.choice(population[:elite_count])
                mutated = self.operator.apply(parent)
                if mutated:
                    m = random.choice(mutated)
                    if self.validator.is_valid(m) and self.budget.validate_edit_distance(m, source):
                        repopulated.append(m)
            population.extend(repopulated)
                    
        return best_candidate
""")

with open("src/materials_adv/attacks/search/llm/proposer.py", "w") as f:
    f.write("""import json
from typing import List, Dict

from materials_adv.framework.interfaces import SearchStrategy, Candidate, Predictor
from materials_adv.framework.objectives import AttackObjective
from materials_adv.framework.budget import Budget
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

class MockLLM:
    def generate(self, prompt: str) -> str:
        # Strictly mocked for parser/schema/validator testing. PROTOCOL_ONLY.
        return json.dumps({
            "operation": "substitution",
            "site": "C",
            "replacement": "N",
            "rationale": "Testing schema."
        })

class LLMProposalStrategy(SearchStrategy):
    def __init__(self, 
                 objective: AttackObjective, 
                 predictor: Predictor, 
                 budget: Budget,
                 llm = None):
        self.objective = objective
        self.predictor = predictor
        self.budget = budget
        self.llm = llm or MockLLM()
        self.validator = RDKitValidityChecker()

    def _parse_llm_output(self, output: str) -> Dict:
        try:
            parsed = json.loads(output)
            required_keys = ["operation", "site", "replacement", "rationale"]
            for k in required_keys:
                if k not in parsed:
                    return None
            return parsed
        except Exception:
            return None

    def search(self, source: Candidate) -> Candidate:
        best_candidate = source
        best_score = float('-inf')
        
        while not self.budget.is_exhausted():
            prompt = "Propose an edit."
            response = self.llm.generate(prompt)
            parsed = self._parse_llm_output(response)
            
            if not parsed:
                continue
                
            # Mock candidate string
            candidate_smiles = source.identifier
            new_c = Candidate(candidate_smiles, provenance=source.provenance + ["LLM_edit"])
            
            if self.validator.is_valid(new_c) and self.budget.validate_edit_distance(new_c, source):
                if not self.budget.use_query():
                    break
                score = self.objective.evaluate(self.predictor, source, new_c)
                if score > best_score:
                    best_score = score
                    best_candidate = new_c
                    
        return best_candidate
""")

