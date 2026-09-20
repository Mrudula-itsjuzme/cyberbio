from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from probabilistic_search_lab.accounting import QueryBudgetExceeded
from probabilistic_search_lab.search_common import begin, finalize


def _pick(families, source, current, rng):
    family = families[int(rng.integers(len(families)))]
    proposals = list(family.propose(source, current, rng))
    if not proposals:
        return None, 0
    return proposals[int(rng.integers(len(proposals)))], len(proposals)


class RandomSearch:
    def __init__(self, families): self.families = list(families)

    def run(self, *, source_id, source, model, validator, budget, seed):
        ledger, target, started = begin(model, budget)
        rng = np.random.default_rng(seed)
        source_pred = target.predict(source)
        effects, valid, invalid, best, best_effect, best_family = [], set(), 0, None, -1.0, None
        attempts = 0
        max_attempts = max(100, budget * 100)
        while ledger.remaining_target and attempts < max_attempts:
            attempts += 1
            proposal, generated = _pick(self.families, source, source, rng)
            ledger.proposal_count += generated
            if proposal is None: break
            ledger.validity_checks += 1
            check = validator.validate(source, proposal.candidate)
            if not check.eligible:
                invalid += 1
                continue
            effect = abs(target.predict(proposal.candidate) - source_pred)
            effects.append(effect); valid.add(proposal.candidate)
            if effect > best_effect:
                best, best_effect, best_family = proposal.candidate, effect, proposal.family
        return finalize(method="random", seed=seed, source_id=source_id, budget=budget,
            ledger=ledger, started=started, effects=effects, valid_candidates=valid,
            invalid_count=invalid, best_candidate=best,
            metadata={"best_edit_type": best_family, "proposal_limit_reached": attempts >= max_attempts})


class EvolutionarySearch:
    def __init__(self, families, population_size=6, elite_fraction=0.34):
        self.families = list(families); self.population_size = population_size
        self.elite_fraction = elite_fraction

    def run(self, *, source_id, source, model, validator, budget, seed):
        ledger, target, started = begin(model, budget); rng = np.random.default_rng(seed)
        source_pred = target.predict(source); population = [(source, 0.0)]
        effects, valid, invalid, best, best_effect, best_family, history = [], set(), 0, None, -1.0, None, []
        while ledger.remaining_target:
            elite_n = max(1, int(np.ceil(len(population) * self.elite_fraction)))
            parents = sorted(population, key=lambda x: x[1], reverse=True)[:elite_n]
            children = []
            attempts = 0
            while len(children) < self.population_size and attempts < self.population_size * 10:
                attempts += 1
                parent = parents[int(rng.integers(len(parents)))][0]
                proposal, generated = _pick(self.families, source, parent, rng)
                ledger.proposal_count += generated
                if proposal is None: continue
                ledger.validity_checks += 1
                if not validator.validate(source, proposal.candidate).eligible:
                    invalid += 1; continue
                children.append((proposal.candidate, proposal.family))
            if not children: break
            population = []
            for candidate, family_name in dict.fromkeys(children):
                if not ledger.remaining_target: break
                effect = abs(target.predict(candidate) - source_pred)
                effects.append(effect); valid.add(candidate); population.append((candidate, effect))
                if effect > best_effect: best, best_effect, best_family = candidate, effect, family_name
            history.append({"generation": len(history), "population": len(population),
                            "best_effect": max((x[1] for x in population), default=None),
                            "queries": ledger.target_model_queries})
            if not population: break
        return finalize(method="evolutionary", seed=seed, source_id=source_id, budget=budget,
            ledger=ledger, started=started, effects=effects, valid_candidates=valid,
            invalid_count=invalid, best_candidate=best, trajectories={"generations": history},
            metadata={"best_edit_type": best_family})


class MCMCSearch:
    """Accounting wrapper around the existing Metropolis-style implementation."""
    def __init__(self, temperature=5.0, min_tanimoto=0.5):
        self.temperature = temperature; self.min_tanimoto = min_tanimoto

    def run(self, *, source_id, source, model, validator, budget, seed):
        ledger, target, started = begin(model, budget); rng = np.random.default_rng(seed)
        materials = Path(__file__).resolve().parents[3] / "materials-adversarial"
        sys.path.insert(0, str(materials / "src"))
        from materials_adv.domain.chemistry.attacks.probabilistic import ProbabilisticMCMCAttack
        import json
        tokens = json.loads((materials / "data/processed/vocab.json").read_text())

        class Proxy:
            def predict(self, sequences): return target.predict_batch(sequences)

        attack = ProbabilisticMCMCAttack(
            rng=rng, predictor=Proxy(), allowed_tokens=tokens,
            steps=max(0, budget - 1), temperature=self.temperature,
            min_tanimoto_similarity=self.min_tanimoto,
        )
        proposal_validity = []
        class CountedProposal:
            def __init__(self, delegate): self.delegate = delegate
            def generate(self, *args, **kwargs):
                generated = self.delegate.generate(*args, **kwargs)
                ledger.proposal_count += len(generated)
                for outcome in generated:
                    ledger.validity_checks += 1
                    proposal_validity.append(
                        (outcome.adversarial_representation,
                         validator.validate(source, outcome.adversarial_representation))
                    )
                return generated
        attack._proposals = [CountedProposal(proposal) for proposal in attack._proposals]
        outcomes = []
        failure = None
        try:
            outcomes = attack.generate(list(source), n_variants=max(1, budget - 1))
        except QueryBudgetExceeded:
            failure = "TARGET_QUERY_BUDGET_EXCEEDED"
        source_pred = target.last_prediction(source)
        validity_by_candidate = {candidate: check for candidate, check in proposal_validity}
        invalid = sum(not check.eligible for _, check in proposal_validity)
        effects, valid, best, best_effect, trajectory = [], set(), None, -1.0, []
        if source_pred is not None:
            for candidate, prediction in target.history[1:]:
                check = validity_by_candidate.get(candidate)
                if check is None:
                    ledger.validity_checks += 1
                    check = validator.validate(source, candidate)
                if not check.eligible:
                    continue
                effect = abs(prediction - source_pred)
                effects.append(effect); valid.add(candidate)
                if effect > best_effect: best, best_effect = candidate, effect
        for outcome in outcomes:
            candidate = outcome.adversarial_representation
            prediction = target.last_prediction(candidate)
            if prediction is None or source_pred is None: continue
            effect = abs(prediction - source_pred)
            trajectory.append({"candidate": candidate, "effect": effect,
                               "acceptance_probability": outcome.params.get("sampling_probability")})
        return finalize(method="mcmc", seed=seed, source_id=source_id, budget=budget,
            ledger=ledger, started=started, effects=effects, valid_candidates=valid,
            invalid_count=invalid, best_candidate=best, trajectories={"accepted": trajectory},
            failure_status=failure, metadata={"canonical_name": "Metropolis-style search",
                                               "best_edit_type": "probabilistic_mcmc" if best else None})
