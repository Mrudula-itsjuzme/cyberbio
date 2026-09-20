"""Black-box search policies with explicit model-query accounting."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from ...data.tokenizer import tokenize
from ...validation.pipeline import validate
from ..generator import PredictorProtocol
from ..token_space import count_changes
from .proposals import Proposal


class ProposalProtocol(Protocol):
    @property
    def names(self) -> tuple[str, ...]: ...
    def propose(self, tokens: Sequence[str]) -> Proposal | None: ...
    def enumerate_proposals(self, tokens: Sequence[str]) -> list[Proposal]: ...

@dataclass
class SearchResult:
    strategy: str
    original_representation: str
    best_representation: str
    original_prediction: float
    best_prediction: float
    best_drift: float
    candidate_query_budget: int
    candidate_model_queries: int
    total_model_queries: int
    candidate_proposals: int
    representation_valid_candidates: int
    valid_candidates: int
    accepted_candidates: int
    acceptance_rate: float | None
    perturbation_size: int
    elapsed_seconds: float
    proposal_operators: tuple[str, ...]
    trace: list[dict] = field(default_factory=list)

    @property
    def drift_per_model_query(self) -> float:
        return self.best_drift / self.candidate_model_queries if self.candidate_model_queries else 0.0

    def summary(self) -> dict:
        result = {key: value for key, value in self.__dict__.items() if key != "trace"}
        result["drift_per_model_query"] = self.drift_per_model_query
        result["query_budget_exhausted"] = self.candidate_model_queries == self.candidate_query_budget
        result["proposal_operators"] = list(self.proposal_operators)
        return result


@dataclass(frozen=True)
class _Candidate:
    tokens: tuple[str, ...]
    representation: str
    prediction: float
    drift: float


class BlackBoxSearch:
    name = "base"

    def __init__(self, predictor: PredictorProtocol, proposal: ProposalProtocol,
                 rng: np.random.Generator, *, query_budget: int,
                 require_plausible: bool = True, max_changes: int | None = None,
                 max_proposal_multiplier: int = 50, proposal_budget: int | None = None):
        if query_budget < 1:
            raise ValueError("query_budget must be positive")
        self.predictor, self.proposal, self.rng = predictor, proposal, rng
        self.query_budget = query_budget
        self.require_plausible = require_plausible
        self.max_changes = max_changes
        self.proposal_budget = proposal_budget if proposal_budget is not None else query_budget * max_proposal_multiplier

    def _run(self, original: _Candidate, evaluate, record):
        raise NotImplementedError

    def search(self, representation: str) -> SearchResult:
        started = time.perf_counter()
        original_tokens = tuple(tokenize(representation))
        original_prediction = float(self.predictor.predict([representation])[0])
        original = _Candidate(original_tokens, representation, original_prediction, 0.0)
        cache = {representation: original}
        counters = {"proposals": 0, "rep_valid": 0, "valid": 0, "queries": 0, "accepted": 0}
        trace: list[dict] = []

        def evaluate(parent: _Candidate, precomputed_proposal: Proposal | None = None) -> tuple[_Candidate | None, Proposal | None, dict]:
            counters["proposals"] += 1
            proposed = precomputed_proposal if precomputed_proposal else self.proposal.propose(parent.tokens)
            entry = {"proposal_index": counters["proposals"], "parent": parent.representation,
                     "operator": None, "candidate": None, "representation_valid": False,
                     "plausible": False, "queried": False, "prediction": None,
                     "absolute_drift": None, "accepted": False}
            if proposed is None:
                entry["rejection_reason"] = "operator_returned_no_candidate"
                return None, None, entry
            text = "".join(proposed.tokens)
            perturbation_size = count_changes(list(original.tokens), list(proposed.tokens))
            entry.update({"operator": proposed.operator, "candidate": text,
                          "perturbation_size": perturbation_size})
            if self.max_changes is not None and perturbation_size > self.max_changes:
                entry["rejection_reason"] = "perturbation_budget_exceeded"
                return None, proposed, entry
            validity = validate(text, check_plausibility=True)
            entry["representation_valid"] = validity.representation_valid is True
            entry["plausible"] = validity.plausible
            if validity.representation_valid is True:
                counters["rep_valid"] += 1
            eligible = validity.representation_valid is True and (
                validity.plausible or not self.require_plausible)
            if not eligible:
                entry["rejection_reason"] = "invalid_or_implausible"
                return None, proposed, entry
            counters["valid"] += 1
            if text in cache:
                entry["rejection_reason"] = "duplicate_cached_candidate"
                return cache[text], proposed, entry
            if counters["queries"] >= self.query_budget:
                entry["rejection_reason"] = "query_budget_exhausted"
                return None, proposed, entry
            prediction = float(self.predictor.predict([text])[0])
            counters["queries"] += 1
            candidate = _Candidate(proposed.tokens, text, prediction,
                                   abs(prediction - original_prediction))
            cache[text] = candidate
            entry.update({"queried": True, "prediction": prediction,
                          "absolute_drift": candidate.drift})
            return candidate, proposed, entry

        def record(entry: dict, accepted: bool = False):
            entry["accepted"] = bool(accepted)
            if accepted:
                counters["accepted"] += 1
            trace.append(entry)

        best = self._run(original, evaluate, record)
        elapsed = time.perf_counter() - started
        applicable = self.name in {"greedy", "metropolis_mcmc"}
        return SearchResult(
            strategy=self.name, original_representation=representation,
            best_representation=best.representation, original_prediction=original_prediction,
            best_prediction=best.prediction, best_drift=best.drift,
            candidate_query_budget=self.query_budget, candidate_model_queries=counters["queries"],
            total_model_queries=counters["queries"] + 1, candidate_proposals=counters["proposals"],
            representation_valid_candidates=counters["rep_valid"], valid_candidates=counters["valid"],
            accepted_candidates=counters["accepted"],
            acceptance_rate=(counters["accepted"] / counters["valid"] if applicable and counters["valid"] else None),
            perturbation_size=count_changes(list(original.tokens), list(best.tokens)),
            elapsed_seconds=elapsed, proposal_operators=self.proposal.names, trace=trace,
        )

    @staticmethod
    def _better(left: _Candidate, right: _Candidate) -> _Candidate:
        return left if left.drift >= right.drift else right


class RandomSearch(BlackBoxSearch):
    name = "random"
    def _run(self, original, evaluate, record):
        best, current, queries = original, original, 0
        failures = 0
        for _ in range(self.proposal_budget):
            candidate, _, entry = evaluate(current)
            record(entry)
            queries += int(entry["queried"])
            if candidate is not None:
                current = candidate
                best = self._better(best, candidate)
                failures = 0
            else:
                failures += 1
            if failures >= 3 or self.rng.random() < 0.1:
                current = original
                failures = 0
            if queries >= self.query_budget:
                break
        return best


class GreedySearch(BlackBoxSearch):
    name = "greedy"
    def _run(self, original, evaluate, record):
        best, current, queries = original, original, 0
        while queries < self.query_budget:
            proposals = self.proposal.enumerate_proposals(current.tokens)
            if not proposals:
                break
            
            step_best = None
            for p in proposals:
                if queries >= self.query_budget:
                    break
                candidate, _, entry = evaluate(current, precomputed_proposal=p)
                record(entry)
                queries += int(entry["queried"])
                if candidate is not None:
                    if step_best is None or candidate.drift > step_best.drift:
                        step_best = candidate
            
            if step_best is not None and step_best.drift > current.drift:
                current = step_best
                best = self._better(best, current)
            else:
                break
                
        return best


class BeamSearch(BlackBoxSearch):
    name = "beam"
    def __init__(self, *args, beam_width: int = 4, branching_factor: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if beam_width < 1 or branching_factor < 1:
            raise ValueError("beam_width and branching_factor must be positive")
        self.beam_width, self.branching_factor = beam_width, branching_factor

    def _run(self, original, evaluate, record):
        beam, best, queries = [original], original, 0
        while queries < self.query_budget:
            generation = []
            for parent in beam:
                for _ in range(self.branching_factor):
                    candidate, _, entry = evaluate(parent)
                    queries += int(entry["queried"])
                    record(entry)
                    if candidate is not None:
                        generation.append(candidate)
                        best = self._better(best, candidate)
                    if queries >= self.query_budget or entry["proposal_index"] >= self.proposal_budget:
                        break
                if queries >= self.query_budget or entry["proposal_index"] >= self.proposal_budget: break
            if not generation:
                if entry["proposal_index"] >= self.proposal_budget: break
                continue
            unique = {candidate.representation: candidate for candidate in beam + generation}
            beam = sorted(unique.values(), key=lambda item: item.drift, reverse=True)[:self.beam_width]
        return best


class MetropolisSearch(BlackBoxSearch):
    """Metropolis-style heuristic; not claimed as formal MH without proposal correction."""
    name = "metropolis_mcmc"
    def __init__(self, *args, temperature: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if temperature <= 0: raise ValueError("temperature must be positive")
        self.temperature = temperature

    def _run(self, original, evaluate, record):
        current = best = original
        queries = 0
        for _ in range(self.proposal_budget):
            candidate, _, entry = evaluate(current)
            queries += int(entry["queried"])
            accepted = False
            if candidate is not None:
                delta = candidate.drift - current.drift
                probability = min(1.0, math.exp(delta / self.temperature))
                entry["acceptance_probability"] = probability
                accepted = bool(self.rng.random() < probability)
                if accepted: current = candidate
                best = self._better(best, candidate)
            record(entry, accepted)
            if queries >= self.query_budget: break
        return best


class EvolutionarySearch(BlackBoxSearch):
    name = "evolutionary"
    def __init__(self, *args, population_size: int = 8, elite_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if population_size < 1 or not 1 <= elite_size <= population_size:
            raise ValueError("invalid population or elite size")
        self.population_size, self.elite_size = population_size, elite_size

    def _run(self, original, evaluate, record):
        population, best, queries = [original], original, 0
        while queries < self.query_budget:
            offspring = []
            for _ in range(self.population_size):
                parent = population[int(self.rng.integers(0, len(population)))]
                candidate, _, entry = evaluate(parent)
                queries += int(entry["queried"])
                record(entry, False)
                if candidate is not None:
                    offspring.append(candidate)
                    best = self._better(best, candidate)
                if queries >= self.query_budget or entry["proposal_index"] >= self.proposal_budget: break
            if not offspring:
                if entry["proposal_index"] >= self.proposal_budget: break
                continue
            unique = {candidate.representation: candidate for candidate in population + offspring}
            elites = sorted(unique.values(), key=lambda item: item.drift, reverse=True)[:self.elite_size]
            population = elites
        return best


class _MetaProposal:
    def __init__(self, proposals: Sequence[ProposalProtocol]):
        self.proposals = proposals

    @property
    def names(self) -> tuple[str, ...]:
        names = []
        for p in self.proposals:
            names.extend(p.names)
        return tuple(names)

    def propose(self, tokens: Sequence[str]) -> Proposal | None:
        return None

    def enumerate_proposals(self, tokens: Sequence[str]) -> list[Proposal]:
        raise NotImplementedError("MetaProposal should not be used directly for proposals.")


class ThompsonSamplingSearch(BlackBoxSearch):
    """
    Adaptive Attacker Portfolio using a Beta-Bernoulli multi-armed bandit.
    """
    name = "thompson"

    def __init__(
        self,
        predictor: PredictorProtocol,
        proposals: Sequence[ProposalProtocol],
        rng: np.random.Generator,
        *,
        query_budget: int,
        require_plausible: bool = True,
        max_changes: int | None = None,
        max_proposal_multiplier: int = 50,
        proposal_budget: int | None = None,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        reward_mode: str = "binary",
        threshold: float = 0.0,
        update_posterior: bool = True
    ):
        if not proposals:
            raise ValueError("ThompsonSamplingSearch requires at least one proposal protocol.")
            
        super().__init__(
            predictor,
            _MetaProposal(proposals),
            rng,
            query_budget=query_budget,
            require_plausible=require_plausible,
            max_changes=max_changes,
            max_proposal_multiplier=max_proposal_multiplier,
            proposal_budget=proposal_budget
        )
        self.proposals = list(proposals)
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.reward_mode = reward_mode
        self.threshold = threshold
        self.update_posterior = update_posterior

    def _run(self, original: _Candidate, evaluate, record) -> _Candidate:
        best, current, queries = original, original, 0
        iteration = 0
        
        # Initialize Beta distribution parameters for each arm
        arms = {i: [self.alpha_prior, self.beta_prior] for i in range(len(self.proposals))}
        
        for _ in range(self.proposal_budget):
            iteration += 1
            # 1. Sample from the Beta posterior for each arm
            samples = [self.rng.beta(arms[i][0], arms[i][1]) for i in range(len(self.proposals))]
            
            # 2. Select the arm with the highest sample
            chosen_arm_idx = int(np.argmax(samples))
            chosen_proposal_protocol = self.proposals[chosen_arm_idx]
            
            alpha_before = float(arms[chosen_arm_idx][0])
            beta_before = float(arms[chosen_arm_idx][1])
            sampled_theta = float(samples[chosen_arm_idx])
            
            # Generate a proposal from the chosen arm
            precomputed = chosen_proposal_protocol.propose(current.tokens)
            
            candidate, proposed, entry = evaluate(current, precomputed_proposal=precomputed)
            queries += int(entry.get("queried", False))
            
            # 3. Calculate Reward
            reward = 0.0
            accepted = False
            
            # Determine validity and drift
            validity = entry.get("representation_valid", False)
            if self.require_plausible:
                validity = validity and entry.get("plausible", False)
                
            drift = candidate.drift if candidate is not None else 0.0
            
            if self.reward_mode == "binary":
                if validity and drift >= self.threshold:
                    reward = 1.0
            elif self.reward_mode == "validity-aware binary":
                if validity:
                    reward = 1.0
                else:
                    reward = 0.0
            elif self.reward_mode == "continuous":
                norm_factor = max(self.threshold, 1e-5)
                if validity:
                    reward = min(1.0, max(0.0, drift / norm_factor))
                else:
                    reward = 0.0
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")
            
            # Standard acceptance logic (greedy improvement)
            if candidate is not None and candidate.drift > current.drift:
                current = candidate
                best = self._better(best, candidate)
                accepted = True

            # 4. Update Posterior
            # 4. Check Duplicate and Update Posterior
            is_duplicate = entry.get("rejection_reason") == "duplicate_cached_candidate"
            entry["duplicate"] = is_duplicate
            
            if self.update_posterior and not is_duplicate:
                arms[chosen_arm_idx][0] += reward
                arms[chosen_arm_idx][1] += (1.0 - reward)
                entry["posterior_updated"] = True
            else:
                entry["posterior_updated"] = False
                if is_duplicate:
                    reward = 0.0 # Force reward to 0 for logging on duplicate
            
            # Record strictly requested diagnostic fields in the trace entry
            entry["iteration"] = iteration
            entry["chosen_arm"] = chosen_arm_idx
            entry["arm_name"] = chosen_proposal_protocol.names[0] if chosen_proposal_protocol.names else "unknown"
            entry["alpha_before"] = alpha_before
            entry["beta_before"] = beta_before
            entry["sampled_theta"] = sampled_theta
            entry["validity"] = bool(validity)
            entry["drift"] = float(drift)
            entry["reward"] = float(reward)
            entry["alpha_after"] = float(arms[chosen_arm_idx][0])
            entry["beta_after"] = float(arms[chosen_arm_idx][1])
            entry["target_query_count"] = queries
            
            record(entry, accepted)
            
            if queries >= self.query_budget:
                break
                
        return best

class UniformSamplingSearch(ThompsonSamplingSearch):
    """
    A baseline uniform controller that wraps the same proposals as ThompsonSamplingSearch
    but never updates its posterior (Beta(1,1)), equivalent to uniform random choice.
    """
    name = "uniform_portfolio"
    
    def __init__(self, *args, **kwargs):
        kwargs["update_posterior"] = False
        super().__init__(*args, **kwargs)


class ContextualThompsonSamplingSearch(BlackBoxSearch):
    """
    Contextual Thompson Sampling Search using Bayesian Linear Regression.
    """
    def __init__(
        self,
        predictor: PredictorProtocol,
        proposals: Sequence[ProposalProtocol],
        rng: np.random.Generator,
        *,
        query_budget: int,
        require_plausible: bool = True,
        max_changes: int | None = None,
        max_proposal_multiplier: int = 50,
        proposal_budget: int | None = None,
        context_features: np.ndarray,
        mu_priors: dict[str, np.ndarray],
        cov_priors: dict[str, np.ndarray],
        noise_variance: float = 1.0,
        reward_mode: str = "binary",
        threshold: float = 0.0,
        update_posterior: bool = True
    ):
        super().__init__(
            predictor=predictor,
            proposal=proposals[0], # Dummy proposal for base init, overriden in _run
            rng=rng,
            query_budget=query_budget,
            require_plausible=require_plausible,
            max_changes=max_changes,
            max_proposal_multiplier=max_proposal_multiplier,
            proposal_budget=proposal_budget
        )
        self.proposals = proposals
        self.context_features = context_features
        self.mu_priors = mu_priors
        self.cov_priors = cov_priors
        self.noise_variance = noise_variance
        self.reward_mode = reward_mode
        self.threshold = threshold
        self.update_posterior = update_posterior
        
    def _run(self, original: _Candidate, evaluate, record) -> _Candidate:
        best, current = original, original
        iteration = 0
        
        # Initialize arms
        arms_state = []
        for p in self.proposals:
            name = p.names[0]
            # Tracking precision matrix Lambda and precision-mean vector xi
            if name in self.mu_priors and name in self.cov_priors:
                mu = self.mu_priors[name].copy()
                cov = self.cov_priors[name].copy()
                Lambda = np.linalg.inv(cov)
                xi = Lambda @ mu
            else:
                dim = len(self.context_features)
                Lambda = np.eye(dim)
                xi = np.zeros(dim)
            arms_state.append({"name": name, "Lambda": Lambda, "xi": xi, "proposal": p})
            
        z = self.context_features
        z_outer = np.outer(z, z) / self.noise_variance
        
        for _ in range(self.proposal_budget):
            iteration += 1
            
            # Sample from posterior and calculate scores
            scores = []
            for arm in arms_state:
                Lambda = arm["Lambda"]
                xi = arm["xi"]
                # cov = inv(Lambda)
                cov = np.linalg.inv(Lambda)
                mu = cov @ xi
                # Sample theta
                try:
                    theta_hat = self.rng.multivariate_normal(mu, cov)
                except np.linalg.LinAlgError:
                    theta_hat = mu
                
                score = np.dot(z, theta_hat)
                scores.append(score)
                
            # Choose arm
            chosen_arm_idx = int(np.argmax(scores))
            chosen_arm = arms_state[chosen_arm_idx]
            
            # Generate a proposal from the chosen arm
            precomputed = chosen_arm["proposal"].propose(current.tokens)
            
            candidate, proposed, entry = evaluate(current, precomputed_proposal=precomputed)
            
            # Reward
            reward = 0.0
            accepted = False
            validity = entry.get("representation_valid", False)
            if self.require_plausible:
                validity = validity and entry.get("plausible", False)
                
            drift = candidate.drift if candidate is not None else 0.0
            
            if self.reward_mode == "binary":
                if validity and drift >= self.threshold:
                    reward = 1.0
                    
            if candidate is not None and candidate.drift > current.drift:
                current = candidate
                best = self._better(best, candidate)
                accepted = True
                
            is_duplicate = entry.get("rejection_reason") == "duplicate_cached_candidate"
            entry["duplicate"] = is_duplicate
            
            if self.update_posterior and not is_duplicate:
                # Bayesian Linear Update
                chosen_arm["Lambda"] += z_outer
                chosen_arm["xi"] += (z * reward) / self.noise_variance
                entry["posterior_updated"] = True
            else:
                entry["posterior_updated"] = False
                if is_duplicate:
                    reward = 0.0
                    
            entry["iteration"] = iteration
            entry["chosen_arm"] = chosen_arm_idx
            entry["chosen_arm_name"] = chosen_arm["name"]
            entry["reward"] = reward
            entry["score"] = float(scores[chosen_arm_idx])
            
            record(entry, accepted)
            
        return best
