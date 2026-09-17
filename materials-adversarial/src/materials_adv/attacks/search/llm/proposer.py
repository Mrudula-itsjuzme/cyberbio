import json
from typing import List, Dict, Optional

from materials_adv.framework.interfaces import SearchStrategy, Candidate, Predictor, ValidityChecker
from materials_adv.framework.objectives import AttackObjective
from materials_adv.framework.budget import Budget

# NOTE: no module-level RDKit import. The chemistry validator default is resolved
# lazily in __init__ so that importing the generic search layer does not pull RDKit into
# sys.modules (docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 19).

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
                 llm = None,
                 validator: Optional[ValidityChecker] = None):
        self.objective = objective
        self.predictor = predictor
        self.budget = budget
        self.llm = llm or MockLLM()
        if validator is None:
            from materials_adv.domain.chemistry.validator import RDKitValidityChecker
            validator = RDKitValidityChecker()
        self.validator = validator

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
