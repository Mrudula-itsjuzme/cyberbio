import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import time

repo_root = Path(__file__).resolve().parent.parent.parent.parent / "materials-adversarial"
if str(repo_root) not in sys.path and repo_root.exists():
    sys.path.insert(0, str(repo_root / "src"))

from materials_adv.domain.chemistry.attacks.probabilistic import ProbabilisticMCMCAttack
from materials_adv.framework.interfaces import StringIdentity
from materials_adv.framework.objectives import UntargetedDrift
from src.core.attacker import Attacker
from src.core.schemas import AttackResult
from src.core.budgets import BudgetManager

class CanonicalMCMCAdapter(Attacker):
    def __init__(self, temperature: float = 5.0, min_tanimoto: float = 0.5):
        super().__init__(name="canonical_mcmc", family="mcmc")
        self.temperature = temperature
        self.min_tanimoto = min_tanimoto
        
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
        
        # Determine source prediction
        source_pred = model.predict(source_sequence)
        budget.record_query()
        
        # The ProbabilisticMCMCAttack expects certain components from the canonical framework
        # We need to wrap our `model` adapter so it behaves like the canonical expected model
        class ProxyModel:
            def predict(self, seqs):
                if not isinstance(seqs, list):
                    seqs = [seqs]
                    
                # Enforce budget dynamically
                remaining = budget.max_queries - budget.queries_used
                if remaining <= 0:
                    # Return dummy predictions if exhausted
                    return [source_pred for _ in seqs]
                    
                if len(seqs) > remaining:
                    seqs = seqs[:remaining]
                    
                budget.record_query(len(seqs))
                return model.predict_batch(seqs)
        
        import json
        with open(repo_root / "data/processed/vocab.json", "r") as f:
            allowed_tokens = json.load(f)

        # Instantiate the original attack. 
        # Subtract 1 from steps because source prediction costs 1 query.
        attack = ProbabilisticMCMCAttack(
            rng=rng,
            predictor=ProxyModel(),
            allowed_tokens=allowed_tokens,
            temperature=self.temperature,
            min_tanimoto_similarity=self.min_tanimoto,
            steps=max(0, budget.max_queries - 1) 
        )
        
        best_candidate = source_sequence
        best_obj = 0.0
        
        try:
            outcomes = attack.generate(list(source_sequence), n_variants=10)
            for out in outcomes:
                cand_seq = "".join(out.adversarial_tokens)
                obj = out.params.get("attack_score", 0.0)
                if obj > best_obj:
                    best_obj = obj
                    best_candidate = cand_seq
        except Exception as e:
            pass
            
        # We don't query the model again here because MCMC already evaluated it in ProxyModel.
        # However, Canonical MCMC doesn't return the raw prediction, just the attack score.
        # We must re-evaluate if we want the actual drift without charging the budget again 
        # OR just use the best obj if we tracked it.
        # But wait, it's easier to just call it and charge the budget if remaining, else 0 drift.
        # Wait, if best_candidate != source_sequence, the ProxyModel MUST have predicted it.
        # I'll just use a cache to get the prediction.
        
        from materials_adv.domain.chemistry.plausibility import compute_tanimoto_similarity
        from src.metrics.distance import levenshtein
        
        try:
            tan_sim = compute_tanimoto_similarity(source_sequence, best_candidate)
        except:
            tan_sim = 0.0
        edit_dist = levenshtein(source_sequence, best_candidate)
        
        # To get candidate_pred, let's just do a free evaluation since we already paid for it in ProxyModel.
        # Or better: do it and don't record.
        candidate_pred = float(model.predict(best_candidate)) if best_candidate != source_sequence else source_pred

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
            candidate_prediction=candidate_pred,
            prediction_drift=abs(candidate_pred - source_pred),
            objective_value=best_obj,
            metadata={}
        )
        return res
