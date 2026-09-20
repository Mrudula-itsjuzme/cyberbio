from __future__ import annotations

from typing import Sequence
import time
import numpy as np

from .contracts import Attacker, CandidateRecord, Defender, SourceSample


class _DefenderPredictor:
    def __init__(self, defender: Defender):
        self.defender = defender
        self.calls = 0

    def predict(self, candidate) -> float:
        self.calls += 1
        value = candidate.identifier if hasattr(candidate, "identifier") else str(candidate)
        return float(self.defender.predict([value])[0])


class BudgetedSearchAttacker(Attacker):
    """Canonical Framework-V2 search with one centralized accounted evaluator."""

    def __init__(self, name: str, strategy_class, *, seed: int, query_budget: int = 25,
                 max_edits: int = 3, population_size: int = 5, elite_size: int = 2):
        self.name, self.strategy_class = name, strategy_class
        self.proposal_budget, self.max_edits = query_budget, max_edits
        self.population_size, self.elite_size = population_size, elite_size
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        self._last_metrics: dict = {}

    def propose(self, samples: Sequence[SourceSample], defender: Defender) -> list[CandidateRecord]:
        self.source_accounting = []
        from materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution import AliphaticCarbonSubstitutionAttack
        from materials_adv.domain.chemistry.identity import CanonicalSmilesIdentity
        from materials_adv.domain.chemistry.plausibility import compute_tanimoto_similarity
        from materials_adv.domain.chemistry.validator import RDKitValidityChecker
        from materials_adv.framework.accounting import AttackEvaluator
        from materials_adv.framework.budget import Budget
        from materials_adv.framework.interfaces import Candidate
        from materials_adv.framework.objectives import UntargetedDrift

        records, total_calls, total_queries, started = [], 0, 0, time.monotonic()
        for source_index, sample in enumerate(samples):
            predictor = _DefenderPredictor(defender)
            source = Candidate(sample.input, provenance=["source"], operator_edits=0)
            budget = Budget(self.proposal_budget, self.max_edits)
            evaluator = AttackEvaluator(predictor, UntargetedDrift(), budget, source,
                                        identity=CanonicalSmilesIdentity())
            validator = RDKitValidityChecker(require_single_component=True,
                require_same_attachment_count_as=sample.input)
            kwargs = {}
            if self.name == "Evolutionary":
                kwargs = {"population_size": self.population_size, "elite_size": self.elite_size}
            strategy = self.strategy_class(
                AliphaticCarbonSubstitutionAttack(validate_children=False), evaluator, validator,
                seed=self.seed + source_index, **kwargs)
            outcome = strategy.search(source)
            invariants = evaluator.invariants()
            if not invariants["one_model_call_per_query"] or not invariants["within_budget"]:
                raise RuntimeError(f"query accounting invariant failed: {invariants}")
            total_calls += predictor.calls; total_queries += source_queries
            for scored in evaluator.trajectory:
                candidate_input = str(scored.representation)
                self.source_accounting.append({"source_id": sample.source_id, "clean_reference_queries": 0, "attack_target_queries": 0, "attack_query_budget": self.proposal_budget, "validator_calls": 0, "proposals_generated": self.proposal_budget})
                records.append(CandidateRecord(
                    source_id=sample.source_id, original_input=sample.input,
                    candidate_input=candidate_input, attack_family=self.name,
                    attack_variant=strategy.name, proposal_step=scored.index,
                    target_query_count=1, score=scored.objective_value,
                    clean_prediction=float(evaluator.source_prediction),
                    challenging_prediction=scored.prediction, drift=scored.drift,
                    true_target=sample.target, edit_distance=scored.operator_edits,
                    metadata={"target_query_index": scored.index,
                        "provenance": scored.provenance,
                        "similarity": compute_tanimoto_similarity(sample.input, candidate_input),
                        "query_budget": self.proposal_budget,
                        "source_query_count": evaluator.source_scorings,
                        "predictor_calls": predictor.calls,
                        "accounting_invariants": invariants,
                        "search_config": outcome.config},
                ))
        self._last_metrics = {"runtime_seconds": time.monotonic() - started,
            "target_queries": total_queries, "model_calls_including_sources": total_calls,
            "population_size": self.population_size if self.name == "Evolutionary" else None,
            "selection_rule": "elitist" if self.name == "Evolutionary" else "metropolis",
            "query_budget_per_source": self.proposal_budget}
        return records

    def get_state(self) -> dict:
        return {"name": self.name, "version": self.version, "seed": self.seed,
                "query_budget": self.proposal_budget, "max_edits": self.max_edits,
                "training_status": "static", **self._last_metrics}

    def get_metrics(self) -> dict:
        return dict(self._last_metrics)


class ExistingAttackAdapter(Attacker):
    """Adapter for the canonical BaseAttack API; the Arena never validates itself."""

    def __init__(self, name: str, factory, *, seed: int, proposal_budget: int = 2):
        self.name = name
        self.factory = factory
        self.seed = seed
        self.proposal_budget = proposal_budget
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.proposed = 0

    def propose(self, samples: Sequence[SourceSample], defender: Defender) -> list[CandidateRecord]:
        self.source_accounting = []
        from materials_adv.data.tokenizer import tokenize

        attack = self.factory(self.rng, defender)
        records: list[CandidateRecord] = []
        self.source_accounting = []
        for sample in samples:
            outcomes = attack.generate(tokenize(sample.input), n_variants=self.proposal_budget)
                        
            source_queries = sum(int(dict(out.params).get("search_candidate_model_queries", 0)) for out in outcomes)
            self.source_accounting.append({"source_id": sample.source_id, "clean_reference_queries": 0, "attack_target_queries": 0, "attack_query_budget": self.proposal_budget, "validator_calls": 0, "proposals_generated": self.proposal_budget})

            for step, outcome in enumerate(outcomes):
                params = dict(outcome.params)
                queries = int(params.get("search_candidate_model_queries", 0))
                self.source_accounting.append({"source_id": sample.source_id, "clean_reference_queries": 0, "attack_target_queries": 0, "attack_query_budget": self.proposal_budget, "validator_calls": 0, "proposals_generated": self.proposal_budget})
                records.append(CandidateRecord(
                    source_id=sample.source_id, original_input=sample.input,
                    candidate_input=outcome.adversarial_representation,
                    attack_family=self.name, attack_variant=outcome.attack_type,
                    proposal_step=step, target_query_count=queries,
                    candidate_attack_query_count=queries,
                    true_target=sample.target, edit_distance=outcome.number_of_changes,
                    metadata=params,
                ))
        self.proposed += len(records)
        return records

    def get_state(self) -> dict:
        return {"name": self.name, "version": self.version, "seed": self.seed,
                "proposal_budget": self.proposal_budget, "proposed": self.proposed,
                "training_status": "static"}


def random_smiles_adapter(seed: int, proposal_budget: int = 2) -> ExistingAttackAdapter:
    from materials_adv.domain.chemistry.attacks.randomization import SmilesRandomizationAttack
    return ExistingAttackAdapter("RandomSMILES", lambda rng, _: SmilesRandomizationAttack(rng), seed=seed, proposal_budget=proposal_budget)


def mcmc_adapter(seed: int, allowed_tokens: list[str], steps: int = 5, proposal_budget: int = 1) -> ExistingAttackAdapter:
    from materials_adv.domain.chemistry.attacks.probabilistic import ProbabilisticMCMCAttack
    return ExistingAttackAdapter(
        "MCMC",
        lambda rng, defender: ProbabilisticMCMCAttack(rng, predictor=defender, allowed_tokens=allowed_tokens, steps=steps),
        seed=seed, proposal_budget=proposal_budget,
    )


def accounted_mcmc_adapter(seed: int, query_budget: int = 25, max_edits: int = 3) -> BudgetedSearchAttacker:
    from materials_adv.framework.search import MetropolisSearch
    return BudgetedSearchAttacker("MCMC", MetropolisSearch, seed=seed,
                                  query_budget=query_budget, max_edits=max_edits)


def evolutionary_adapter(seed: int, query_budget: int = 25, max_edits: int = 3,
                         population_size: int = 5, elite_size: int = 2) -> BudgetedSearchAttacker:
    from materials_adv.framework.search import EvolutionarySearch
    return BudgetedSearchAttacker("Evolutionary", EvolutionarySearch, seed=seed,
        query_budget=query_budget, max_edits=max_edits,
        population_size=population_size, elite_size=elite_size)
