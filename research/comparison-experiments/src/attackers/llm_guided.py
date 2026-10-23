import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent.parent / "materials-adversarial"
if str(repo_root) not in sys.path and repo_root.exists():
    sys.path.insert(0, str(repo_root / "src"))

from src.core.attacker import Attacker
from src.core.schemas import AttackResult
from src.core.budgets import BudgetManager
from src.attackers.llm_providers import LLMProvider, MockLLMProvider
from src.metrics.distance import levenshtein
from materials_adv.domain.chemistry.plausibility import compute_tanimoto_similarity

class LLMGuidedAttacker(Attacker):
    def __init__(self, provider: LLMProvider, prompt_mode: str = "blind", max_proposals_per_call: int = 5):
        super().__init__(name="llm_guided_search", family="llm")
        self.provider = provider
        self.prompt_mode = prompt_mode
        self.max_proposals = max_proposals_per_call

    def _build_prompt(self, source_sequence: str, history: List[Dict[str, Any]]) -> str:
        # Load from file
        prompt_file = Path(__file__).resolve().parent.parent / f"configs/prompts/{self.prompt_mode}.txt"
        if not prompt_file.exists():
            template = "Source sequence (PSMILES): {source_sequence}\nPropose diverse valid candidate PSMILES sequences by mutating the source sequence.\n"
            self._current_prompt_hash = "fallback_hash"
        else:
            with open(prompt_file, "r") as f:
                template = f.read()
            self._current_prompt_hash = hashlib.sha256(template.encode('utf-8')).hexdigest()

        history_str = ""
        if history:
            for h in history[-3:]: # Show last 3
                history_str += f"- Candidate: {h['cand']}, Valid: {h['valid']}, Drift: {h['drift']:.4f}\n"
        else:
            history_str = "No previous attempts yet.\n"

        prompt = template.replace("{source_sequence}", source_sequence).replace("{history}", history_str)
        prompt += f"\nGenerate up to {self.max_proposals} proposals."
        return prompt

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
        
        # Determine actual seed passed to the generator to log exactly

        budget.start()
        source_pred = float(model.predict(source_sequence))
        budget.record_query()
        
        best_candidate = source_sequence
        best_pred = source_pred
        best_drift = 0.0
        
        total_llm_calls = 0
        total_in_tokens = 0
        total_out_tokens = 0
        failure_reason = ""
        
        # Specific LLM real metrics
        total_proposals = 0
        parsed_proposals = 0
        unique_proposals = 0
        rdkit_valid_proposals = 0
        constraint_pass_proposals = 0
        evaluated_proposals = 0
        successful_proposals = 0
        duplicate_proposals = 0
        
        seen = set([source_sequence])
        history = []
        
        while not budget.is_exhausted():
            prompt = self._build_prompt(source_sequence, history)
            seed_val = int(rng.integers(0, 1000000))
            
            response = self.provider.generate(prompt, max_proposals=self.max_proposals, seed=seed_val)
            
            total_llm_calls += response.get("calls", 1)
            total_in_tokens += response.get("input_tokens", 0)
            total_out_tokens += response.get("output_tokens", 0)
            
            # Use fixed provider error tag, put message in metadata
            if "error" in response:
                failure_reason = "provider_error"
                if metadata is None:
                    metadata = {}
                metadata["provider_error_msg"] = response["error"]
                break
                
            try:
                data = json.loads(response["content"])
                proposals = data.get("proposals", [])
                if not isinstance(proposals, list):
                    proposals = []
            except json.JSONDecodeError:
                proposals = []
                failure_reason = "malformed_json"
                
            if not proposals:
                failure_reason = failure_reason or "empty_response"
                budget.record_generation(1) # Prevent infinite loop on empty
                continue
                
            budget.record_generation(len(proposals))
            total_proposals += len(proposals)
            
            for p in proposals:
                cand = p.get("sequence", "")
                if not cand:
                    continue
                
                parsed_proposals += 1
                rationale = p.get("rationale", "")
                
                if cand in seen:
                    duplicate_proposals += 1
                    history.append({"cand": cand, "valid": False, "drift": 0.0, "rationale": rationale})
                    continue
                    
                seen.add(cand)
                unique_proposals += 1
                
                # Validation checks
                is_valid = validator.is_valid_rdkit(cand)
                if is_valid:
                    rdkit_valid_proposals += 1
                    
                is_plausible = validator.is_valid_plausible(cand)
                if is_plausible:
                    constraint_pass_proposals += 1
                else:
                    history.append({"cand": cand, "valid": False, "drift": 0.0, "rationale": rationale})
                    continue
                
                if budget.is_exhausted():
                    break
                    
                pred = float(model.predict(cand))
                budget.record_query()
                evaluated_proposals += 1
                
                drift = abs(pred - source_pred)
                history.append({"cand": cand, "valid": True, "drift": drift, "rationale": rationale})
                
                if drift > 0.0:
                    successful_proposals += 1
                    
                if drift > best_drift:
                    best_drift = drift
                    best_candidate = cand
                    best_pred = pred
                    
        # If we exhausted without finding anything and no prior failure set
        if not failure_reason and best_drift == 0.0:
            if budget.is_exhausted():
                failure_reason = "budget_exhausted"
            else:
                failure_reason = "no_improvement"
                    
        # Find best rationale if any
        best_rationale = ""
        for h in history:
            if h["cand"] == best_candidate:
                best_rationale = h.get("rationale", "")
                break
                
        # Real Tanimoto & Levenshtein
        try:
            tan_sim = compute_tanimoto_similarity(source_sequence, best_candidate)
            if tan_sim is None:
                tan_sim = 0.0
        except Exception:
            tan_sim = 0.0
            
        edit_dist = levenshtein(source_sequence, best_candidate)
                
        # Combine incoming metadata
        res_metadata = metadata or {}
        model_name = getattr(self.provider, "model_name", getattr(self.provider, "model", "UNAVAILABLE"))
        res_metadata.update({
            "history": history, 
            "rationale": best_rationale, 
            "prompt_hash": getattr(self, "_current_prompt_hash", "none"),
            "provider": self.provider.__class__.__name__,
            "model": model_name
        })

        res = AttackResult(
            source_id=source_id,
            source_sequence=source_sequence,
            candidate_sequence=best_candidate,
            attack_name=f"{self.name}_{self.prompt_mode}",
            attack_family=self.family,
            seed=seed, 
            query_count=budget.queries_used,
            generation_count=budget.generations_used,
            runtime_seconds=budget.runtime_seconds,
            valid_rdkit=validator.is_valid_rdkit(best_candidate),
            constraint_pass=validator.is_valid_plausible(best_candidate),
            tanimoto_similarity=tan_sim, 
            edit_distance=edit_dist, 
            source_prediction=source_pred,
            candidate_prediction=best_pred,
            prediction_drift=best_drift,
            objective_value=best_drift,
            duplicate_proposals=duplicate_proposals,
            failure_reason=failure_reason,
            llm_calls=total_llm_calls,
            llm_input_tokens=total_in_tokens,
            llm_output_tokens=total_out_tokens,
            prompt_mode=self.prompt_mode,
            total_proposals=total_proposals,
            parsed_proposals=parsed_proposals,
            unique_proposals=unique_proposals,
            rdkit_valid_proposals=rdkit_valid_proposals,
            constraint_pass_proposals=constraint_pass_proposals,
            evaluated_proposals=evaluated_proposals,
            successful_proposals=successful_proposals,
            metadata=res_metadata
        )
        return res
