from __future__ import annotations

import time
import numpy as np
from typing import Sequence, Any

from arena.contracts import Attacker, CandidateRecord, Defender, SourceSample
from probabilistic_search_lab.interfaces import TargetModel, Validator, Validation, ProposalFamily, Proposal
from probabilistic_search_lab.canonical import CanonicalValidator
from probabilistic_search_lab.proposals import canonical_families

from cem.search import CEMSearch
from smc.search import SMCSearch
from validity_smc.model import EmpiricalValidityModel

class _ArenaPredictor(TargetModel):
    def __init__(self, defender: Defender, source_input: str):
        self.defender = defender
        self.source_input = source_input
        # Compute clean reference prediction once
        self.clean_pred = float(self.defender.predict([source_input])[0])
        self.calls = 0
        self.clean_reference_queries = 1
        self.queries_log = []

    def predict(self, sequence: str) -> float:
        if sequence == self.source_input:
            # Do not count clean reference queries against the attack budget
            return self.clean_pred
            
        self.calls += 1
        pred = float(self.defender.predict([sequence])[0])
        self.queries_log.append((sequence, pred))
        return pred

class _ArenaValidator(Validator):
    def __init__(self):
        self.canonical = CanonicalValidator()
        self.calls = 0
        self.validation_log = {}

    def validate(self, source: str, candidate: str) -> Validation:
        self.calls += 1
        res = self.canonical.validate(source, candidate)
        self.validation_log[candidate] = res
        return res

class _ArenaFamilyWrapper(ProposalFamily):
    def __init__(self, family: ProposalFamily, tracker: list):
        self.family = family
        self.name = family.name
        self.tracker = tracker

    def propose(self, source: str, current: str, rng) -> Sequence[Proposal]:
        proposals = self.family.propose(source, current, rng)
        for p in proposals:
            self.tracker.append(p)
        return proposals

class ProbabilisticAdapter(Attacker):
    def __init__(self, name: str, search_method_class, seed: int, query_budget: int = 25, state_mode: str = "reset", **kwargs):
        self.name = name
        self.search_method_class = search_method_class
        self.seed = seed
        self.query_budget = query_budget
        self.state_mode = state_mode
        self.kwargs = kwargs
        self._last_metrics = {}
        # Instantiate persistent state if needed
        self.validity_model = None
        if name == "Validity-SMC":
            self.validity_model = EmpiricalValidityModel()
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.proposed = 0
        if self.state_mode == "reset" and self.name == "Validity-SMC":
            from validity_smc.model import EmpiricalValidityModel
            self.validity_model = EmpiricalValidityModel()

    def propose(self, samples: Sequence[SourceSample], defender: Defender) -> list[CandidateRecord]:
        records = []
        started = time.monotonic()
        total_queries = 0
        
        # Aggregation for internal metrics
        all_family_weights = []
        all_ess = []
        all_resamples = []
        self.source_accounting = []

        for idx, sample in enumerate(samples):
            run_seed = self.seed + idx
            
            model = _ArenaPredictor(defender, sample.input)
            validator = _ArenaValidator()
            
            source_pred = model.clean_pred
            
            proposals_tracker = []
            families = [_ArenaFamilyWrapper(f, proposals_tracker) for f in canonical_families()]
            
            kwargs = dict(self.kwargs)
            if self.name == "Validity-SMC":
                kwargs["validity_model"] = self.validity_model
                
            search_instance = self.search_method_class(families=families, **kwargs)
            
            result = search_instance.run(
                source_id=sample.source_id,
                source=sample.input,
                model=model,
                validator=validator,
                budget=self.query_budget,
                seed=run_seed
            )
            
            if "family_weights" in result.metadata:
                all_family_weights.append(result.metadata["family_weights"])
                
            if "smc" in result.trajectories:
                traj = result.trajectories["smc"]
                if traj:
                    all_ess.append(np.mean([step["ess"] for step in traj]))
                    all_resamples.append(sum(1 for step in traj if step.get("resampled")))
            
            
            source_target_queries = model.calls
            source_validator_calls = validator.calls
            
            self.source_accounting.append({
                "source_id": sample.source_id,
                "clean_reference_queries": 1,
                "attack_target_queries": source_target_queries,
                "attack_query_budget": self.query_budget,
                "validator_calls": source_validator_calls,
                "proposals_generated": len(proposals_tracker)
            })

            cand_to_prop = {p.candidate: p for p in proposals_tracker}
            
            for query_idx, (cand_str, pred) in enumerate(model.queries_log):
                if cand_str == sample.input:
                    continue
                
                prop = cand_to_prop.get(cand_str)
                family_name = prop.family if prop else "unknown"
                edit_cost = int(prop.edit_cost) if prop else 1
                
                drift = abs(pred - source_pred)
                val_res = validator.validation_log.get(cand_str)
                
                records.append(CandidateRecord(
                    source_id=sample.source_id,
                    original_input=sample.input,
                    candidate_input=cand_str,
                    attack_family=self.name,
                    attack_variant=self.name,
                    proposal_step=query_idx,
                    validity_status="valid" if val_res and val_res.eligible else "invalid",
                    target_query_count=1,
                    candidate_attack_query_count=1,
                    score=drift,
                    clean_prediction=source_pred,
                    challenging_prediction=pred,
                    drift=drift,
                    true_target=sample.target,
                    edit_distance=edit_cost,
                    metadata={
                        "target_query_index": query_idx,
                        "query_budget": self.query_budget,
                        "trajectories": result.trajectories,
                        "search_metadata": result.metadata
                    }
                ))
            
            total_queries += model.calls

        self._last_metrics = {
            "runtime_seconds": time.monotonic() - started,
            "target_queries": total_queries,
            "clean_reference_queries": len(samples),
            "query_budget_per_source": self.query_budget
        }
        if all_family_weights:
            self._last_metrics["mean_family_weights"] = np.mean(all_family_weights, axis=0).tolist()
        if all_ess:
            self._last_metrics["mean_ess"] = float(np.mean(all_ess))
            self._last_metrics["total_resamples"] = int(np.sum(all_resamples))
            
        self.proposed += len(records)
        return records

    def save_state(self, path: str):
        import json
        if self.name == "Validity-SMC" and self.state_mode == "carry":
            with open(path, "w") as f:
                json.dump(self.validity_model.snapshot(), f)
        else:
            with open(path, "w") as f:
                json.dump({"NO_PERSISTENT_ADAPTATION_STATE": True}, f)

    def load_state(self, path: str):
        import json
        import os
        if not os.path.exists(path): return
        if self.name == "Validity-SMC" and self.state_mode == "carry":
            with open(path, "r") as f:
                data = json.load(f)
            from validity_smc.model import EmpiricalValidityModel
            self.validity_model = EmpiricalValidityModel()
            self.validity_model.restore(data)

    def get_state(self) -> dict:
        state = {"name": self.name, "version": self.version, "seed": self.seed,
                "query_budget": self.query_budget, "proposed": self.proposed,
                "training_status": "probabilistic", **self._last_metrics}
        if self.validity_model:
            state["validity_posterior"] = self.validity_model.snapshot()
        return state


def cem_adapter(seed: int, query_budget: int = 25, state_mode: str = 'reset') -> ProbabilisticAdapter:
    return ProbabilisticAdapter("CEM", CEMSearch, seed=seed, query_budget=query_budget, state_mode=state_mode, batch_size=8, elite_fraction=0.25)

def smc_adapter(seed: int, query_budget: int = 25, state_mode: str = 'reset') -> ProbabilisticAdapter:
    return ProbabilisticAdapter("SMC", SMCSearch, seed=seed, query_budget=query_budget, state_mode=state_mode, particles=8)

def validity_smc_adapter(seed: int, query_budget: int = 25, state_mode: str = 'reset') -> ProbabilisticAdapter:
    return ProbabilisticAdapter("Validity-SMC", SMCSearch, seed=seed, query_budget=query_budget, state_mode=state_mode, particles=8)
