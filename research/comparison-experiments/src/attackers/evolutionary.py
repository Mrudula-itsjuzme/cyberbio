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
from materials_adv.framework.interfaces import Candidate

class EvolutionaryAttacker(Attacker):
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.5, elitism_count: int = 1, early_stopping_patience: int = 50):
        super().__init__(name="evolutionary_search", family="evolutionary")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.early_stopping_patience = early_stopping_patience
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
        source_pred = float(model.predict(source_sequence))
        budget.record_query()
        
        # Initialize population
        population = [source_sequence]
        fitness = [0.0]
        
        best_candidate = source_sequence
        best_pred = source_pred
        best_drift = 0.0
        
        duplicate_proposals = 0
        seen = set([source_sequence])
        
        generation_history = []
        patience_counter = 0
        
        while not budget.is_exhausted():
            new_population = []
            
            # Elitism
            if len(population) > 0 and self.elitism_count > 0:
                best_indices = np.argsort(fitness)[-self.elitism_count:]
                for i in reversed(best_indices):
                    new_population.append(population[i])
            
            gen_valid_count = 0
            gen_total_count = 0
            
            while len(new_population) < self.population_size and not budget.is_exhausted():
                idx1, idx2 = rng.choice(len(population), size=2, replace=True)
                parent = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
                
                child = parent
                if rng.random() < self.mutation_rate:
                    op = rng.choice(self.operators)
                    cand_obj = Candidate(parent)
                    candidates = op.apply(cand_obj)
                    budget.record_generation(len(candidates))
                    if candidates:
                        cand_idx = rng.integers(len(candidates))
                        child = candidates[cand_idx].identifier
                        
                gen_total_count += 1
                if child in seen:
                    duplicate_proposals += 1
                else:
                    seen.add(child)
                    
                if validator.is_valid_plausible(child):
                    new_population.append(child)
                    gen_valid_count += 1
                    
            if budget.is_exhausted():
                break
                
            population = new_population
            fitness = []
            
            for ind in population:
                if budget.is_exhausted():
                    fitness.append(0.0)
                    continue
                pred = float(model.predict(ind))
                budget.record_query()
                drift = abs(pred - source_pred)
                fitness.append(drift)
                
                if drift > best_drift:
                    best_drift = drift
                    best_candidate = ind
                    best_pred = pred
                    patience_counter = 0
            
            patience_counter += 1
            unique_in_pop = len(set(population))
            
            generation_history.append({
                "best_drift": float(np.max(fitness)) if fitness else 0.0,
                "mean_fitness": float(np.mean(fitness)) if fitness else 0.0,
                "validity_rate": gen_valid_count / gen_total_count if gen_total_count > 0 else 0.0,
                "diversity": unique_in_pop / len(population) if population else 0.0,
                "queries_used": budget.queries_used
            })
            
            if patience_counter >= self.early_stopping_patience:
                break
                
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
            metadata={"generation_history": generation_history}
        )
        return res
