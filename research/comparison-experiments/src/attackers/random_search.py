import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import time

repo_root = Path(__file__).resolve().parent.parent.parent.parent / "materials-adversarial"
if str(repo_root) not in sys.path and repo_root.exists():
    sys.path.insert(0, str(repo_root / "src"))

from src.core.attacker import Attacker
from src.core.schemas import AttackResult
from src.core.budgets import BudgetManager
from materials_adv.domain.chemistry.attacks.simple_substitution import SimpleSubstitutionAttack
from materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution import AliphaticCarbonSubstitutionAttack
from materials_adv.framework.interfaces import Candidate, StringIdentity

class RandomSearchAttacker(Attacker):
    def __init__(self):
        super().__init__(name="random_search", family="random")
        self.operators = [
            SimpleSubstitutionAttack(),
            AliphaticCarbonSubstitutionAttack()
        ]

    def attack(
        self,
        source_id: str,
        source_sequence: str,
        model: Any,
        validator: Any,
        budget: BudgetManager,
        rng: np.random.Generator,
        seed: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttackResult:
        
        budget.start()
        source_pred = model.predict(source_sequence)
        budget.record_query()
        
        best_candidate = source_sequence
        best_pred = source_pred
        best_drift = 0.0
        
        while not budget.is_exhausted():
            # Pick random operator
            op = rng.choice(self.operators)
            cand_obj = Candidate(best_candidate)
            candidates = op.apply(cand_obj)
            budget.record_generation(len(candidates))
            
            if not candidates:
                continue
                
            cand_idx = rng.integers(len(candidates))
            cand = candidates[cand_idx].identifier
            
            # Validation
            if not validator.is_valid_plausible(cand):
                continue
                
            if budget.is_exhausted():
                break
                
            pred = model.predict(cand)
            budget.record_query()
            
            drift = abs(pred - source_pred)
            if drift > best_drift:
                best_drift = drift
                best_candidate = cand
                best_pred = pred
                
        from materials_adv.domain.chemistry.plausibility import compute_tanimoto_similarity
        from src.metrics.distance import levenshtein
        try:
            tan_sim = compute_tanimoto_similarity(source_sequence, best_candidate)
        except:
            tan_sim = 0.0
        edit_dist = levenshtein(source_sequence, best_candidate)

        res = AttackResult(
            source_id=source_id,
            source_sequence=source_sequence,
            candidate_sequence=best_candidate,
            attack_name=self.name,
            attack_family=self.family,
            seed=seed, 
            query_count=budget.queries_used,
            generation_count=budget.generations_used,
            runtime_seconds=budget.runtime_seconds,
            valid_rdkit=validator.is_valid_rdkit(best_candidate),
            constraint_pass=validator.is_valid_plausible(best_candidate),
            tanimoto_similarity=tan_sim if tan_sim is not None else 0.0, 
            edit_distance=edit_dist, 
            source_prediction=source_pred,
            candidate_prediction=best_pred,
            prediction_drift=best_drift,
            objective_value=best_drift,
            duplicate_proposals=0,
            failure_reason="",
            metadata={}
        )
        return res
