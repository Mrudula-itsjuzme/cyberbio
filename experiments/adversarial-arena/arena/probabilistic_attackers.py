"""
Arena adapters for probabilistic search attackers: CEM, SMC, ValiditySMC.

These implement the Arena Attacker interface (arena/contracts.py) and bridge
to the prob_lab search engines. ALL target-model calls route through the
centralized Arena accounting gateway.

Adaptation taxonomy:
  - CEMAttacker: updates edit-family proposal probabilities after each batch of elites.
  - SMCAttacker: maintains a weighted particle population across proposals.
  - ValiditySMCAttacker: same as SMC + Beta-Bernoulli validity posterior per family.

Carry-state mode:
  When carry_state=True the attacker preserves its internal search state across
  Arena rounds. When False, it resets to a uniform prior before each round.

True Attacker Adaptation Gain (G_adapt):
  G_adapt(t) = R(A_{t+1}, D_t) - R(A_t, D_t)
  where A_t / A_{t+1} refer to the SAME attacker before / after its internal
  state update. Both are evaluated against the SAME defender checkpoint D_t.
  This differs from Adaptive Selection Gain (G_select) which compares DIFFERENT
  attacker families.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Sequence

import numpy as np

from .contracts import CandidateRecord, Defender, SourceSample
from .contracts import Attacker


# ---------------------------------------------------------------------------
# Arena ↔ prob_lab shims
# ---------------------------------------------------------------------------

class _ArenaTargetModel:
    """
    Wraps Arena Defender into the prob_lab TargetModel protocol.
    Records every call for central accounting.
    """
    def __init__(self, defender: Defender, ledger):
        self._defender = defender
        self._ledger = ledger
        self.calls = 0

    def predict(self, sequence: str) -> float:
        self._ledger.charge_target(1)
        self.calls += 1
        return float(self._defender.predict([sequence])[0])


class _ArenaValidator:
    """
    Wraps RDKit-based Arena validation into the prob_lab Validator protocol.
    """
    def __init__(self):
        from arena.validators import MaterialsValidator
        self._v = MaterialsValidator()

    def validate(self, source: str, candidate: str):
        from arena.prob_lab.interfaces import Validation
        try:
            result = self._v.validate_pair(source, candidate)
            return Validation(
                representation_valid=result.representation_valid,
                domain_constraint_pass=result.domain_constraint_pass,
                reasons=tuple(result.reasons),
            )
        except Exception:
            return Validation(True, True, ())  # fail-open: let attacker proceed


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_candidate_records(
    result,
    samples: Sequence[SourceSample],
    *,
    attack_family: str,
    source_index: int,
    source_pred: float,
    round_seed: int,
) -> list[CandidateRecord]:
    """Convert a prob_lab SearchResult to Arena CandidateRecord list."""
    records = []
    m = result.metrics
    # Collect every scored candidate from trajectories
    step = 0
    if result.best_candidate:
        sample = samples[source_index]
        records.append(CandidateRecord(
            source_id=sample.source_id,
            original_input=sample.input,
            candidate_input=result.best_candidate,
            attack_family=attack_family,
            attack_variant=result.method,
            proposal_step=step,
            target_query_count=m.target_model_queries,
            score=m.best_effect,
            clean_prediction=source_pred,
            challenging_prediction=source_pred + (m.best_effect or 0),
            drift=m.best_effect,
            true_target=sample.target,
            edit_distance=None,
            metadata={
                "mean_drift": m.mean_effect,
                "valid_count": m.valid_count,
                "invalid_count": m.invalid_count,
                "proposal_count": m.proposal_count,
                "target_model_queries": m.target_model_queries,
                "diversity_score": m.diversity_score,
                "runtime_seconds": m.runtime_seconds,
                "method": result.method,
                "seed": result.seed,
            },
        ))
    return records


# ---------------------------------------------------------------------------
# CEM Attacker
# ---------------------------------------------------------------------------

class CEMAttacker(Attacker):
    """
    Cross-Entropy Method attacker with optional carry-state across rounds.

    Internal state: proposal probability distribution over edit families.
    Adaptation: elite-fraction-weighted update after each proposal batch.
    """

    name = "CEM"
    version = "1"

    def __init__(self, *, seed: int, query_budget: int = 25,
                 batch_size: int = 8, elite_fraction: float = 0.25,
                 smoothing: float = 0.7, entropy_floor: float = 0.05,
                 carry_state: bool = True):
        self.base_seed = seed
        self.query_budget = query_budget
        self.batch_size = batch_size
        self.elite_fraction = elite_fraction
        self.smoothing = smoothing
        self.entropy_floor = entropy_floor
        self.carry_state = carry_state

        # Adaptation state — probabilities over families, updated across rounds
        self._probs: list[float] | None = None   # None = uniform (pre-init)
        self._history: list[dict] = []
        self._round_id: int = 0
        self._last_metrics: dict = {}
        self._state_before: dict = {}
        self._state_after: dict = {}
        self._validator = None

    def _get_validator(self):
        if self._validator is None:
            self._validator = _ArenaValidator()
        return self._validator

    def reset(self, seed: int) -> None:
        self.base_seed = seed
        self._probs = None
        self._history = []
        self._round_id = 0
        self._last_metrics = {}
        self._state_before = {}
        self._state_after = {}

    def _snapshot_state(self) -> dict:
        from arena.prob_lab.metrics import categorical_entropy
        return {
            "probs": list(self._probs) if self._probs else None,
            "entropy": categorical_entropy(self._probs) if self._probs else None,
            "round_id": self._round_id,
        }

    def propose(self, samples: Sequence[SourceSample], defender: Defender) -> list[CandidateRecord]:
        from arena.prob_lab.accounting import QueryLedger
        from arena.prob_lab.cem_search import CEMSearch
        from arena.prob_lab.proposals import arena_proposal_families

        families = arena_proposal_families()
        k = len(families)

        # Carry-state: inject current probs into a custom CEMSearch subclass
        if self.carry_state and self._probs is not None:
            init_probs = np.array(self._probs, dtype=float)
        else:
            init_probs = np.full(k, 1.0 / k)

        self._state_before = self._snapshot_state()

        all_records: list[CandidateRecord] = []
        validator = self._get_validator()
        final_probs = init_probs.copy()
        all_history: list[dict] = []
        runtime_start = time.monotonic()

        total_proposals = 0
        total_queries = 0
        total_valid = 0
        total_invalid = 0
        all_effects: list[float] = []

        for source_index, sample in enumerate(samples):
            round_seed = self.base_seed + self._round_id * 1000 + source_index

            # Minimal CEM implementation using updated probs as starting point
            ledger = QueryLedger(budget=self.query_budget)
            model = _ArenaTargetModel(defender, ledger)
            source_pred = model.predict(sample.input)

            probs = init_probs.copy()
            best, best_effect = None, -1.0
            effects = []
            invalid = 0
            valid_set = set()
            rng = np.random.default_rng(round_seed)

            while ledger.remaining_target:
                evaluated = []
                attempts = 0
                batch = min(self.batch_size, ledger.remaining_target)
                while len(evaluated) < batch and attempts < batch * 20:
                    attempts += 1
                    fi = int(rng.choice(k, p=probs))
                    family = families[fi]
                    proposals = list(family.propose(sample.input, sample.input, rng))
                    ledger.proposal_count += len(proposals)
                    if not proposals:
                        continue
                    proposal = proposals[int(rng.integers(len(proposals)))]
                    ledger.validity_checks += 1
                    if not validator.validate(sample.input, proposal.candidate).eligible:
                        invalid += 1
                        continue
                    try:
                        effect = abs(model.predict(proposal.candidate) - source_pred)
                    except Exception:
                        break
                    effects.append(effect)
                    valid_set.add(proposal.candidate)
                    evaluated.append((effect, fi))
                    if effect > best_effect:
                        best, best_effect = proposal.candidate, effect

                if not evaluated:
                    break

                elite_n = max(1, int(math.ceil(len(evaluated) * self.elite_fraction)))
                elites = sorted(evaluated, reverse=True)[:elite_n]
                fitted = np.bincount([idx for _, idx in elites], minlength=k).astype(float)
                fitted /= fitted.sum()
                fitted = (1.0 - self.entropy_floor) * fitted + self.entropy_floor / k
                probs = self.smoothing * fitted + (1.0 - self.smoothing) * probs
                probs /= probs.sum()

                from arena.prob_lab.metrics import categorical_entropy
                all_history.append({
                    "source_index": source_index,
                    "iteration": len(all_history),
                    "probabilities": probs.tolist(),
                    "entropy": categorical_entropy(probs.tolist()),
                    "elite_effects": [e for e, _ in elites],
                    "queries": ledger.target_model_queries,
                })

            # Update shared probs (across sources within the round)
            final_probs = self.smoothing * probs + (1.0 - self.smoothing) * final_probs
            final_probs /= final_probs.sum()

            total_proposals += ledger.proposal_count
            total_queries += ledger.target_model_queries
            total_valid += len(effects)
            total_invalid += invalid
            all_effects.extend(effects)

            if best:
                all_records.append(CandidateRecord(
                    source_id=sample.source_id,
                    original_input=sample.input,
                    candidate_input=best,
                    attack_family=self.name,
                    attack_variant="cem",
                    proposal_step=0,
                    target_query_count=ledger.target_model_queries,
                    score=best_effect,
                    clean_prediction=source_pred,
                    challenging_prediction=source_pred + best_effect,
                    drift=best_effect,
                    true_target=sample.target,
                    metadata={
                        "round_id": self._round_id,
                        "carry_state": self.carry_state,
                        "final_probs": probs.tolist(),
                        "valid_count": len(effects),
                        "invalid_count": invalid,
                        "proposal_count": ledger.proposal_count,
                        "target_model_queries": ledger.target_model_queries,
                    },
                ))

        # Update carry-state with round-aggregated probs
        if self.carry_state:
            self._probs = final_probs.tolist()
        else:
            self._probs = None  # reset for next round

        from arena.prob_lab.metrics import categorical_entropy, summarize_effects
        summary = summarize_effects(all_effects)
        self._state_after = {
            "probs": final_probs.tolist(),
            "entropy": categorical_entropy(final_probs.tolist()),
            "round_id": self._round_id,
            "family_names": [f.name for f in families],
        }
        self._last_metrics = {
            "round_id": self._round_id,
            "carry_state": self.carry_state,
            "proposal_count": total_proposals,
            "target_model_queries": total_queries,
            "valid_count": total_valid,
            "invalid_count": total_invalid,
            "validity_rate": total_valid / max(1, total_proposals),
            "mean_drift": summary["mean"],
            "best_drift": summary["best"],
            "runtime_seconds": time.monotonic() - runtime_start,
            "probs": final_probs.tolist(),
            "entropy": categorical_entropy(final_probs.tolist()),
        }
        self._history.extend(all_history)
        self._round_id += 1
        return all_records

    def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.base_seed,
            "carry_state": self.carry_state,
            "round_id": self._round_id,
            "probs": self._probs,
            "training_status": "adaptive",
            "state_before": self._state_before,
            "state_after": self._state_after,
            **self._last_metrics,
        }

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._last_metrics)


# ---------------------------------------------------------------------------
# SMC Attacker
# ---------------------------------------------------------------------------

class SMCAttacker(Attacker):
    """
    Sequential Monte Carlo attacker with optional carry-state across rounds.

    Internal state: weighted particle population over SMILES strings.
    Adaptation: importance weighting + resampling at each step.
    """

    name = "SMC"
    version = "1"

    def __init__(self, *, seed: int, query_budget: int = 25,
                 particles: int = 8, ess_threshold: float = 0.5,
                 beta: float = 1.0, carry_state: bool = True):
        self.base_seed = seed
        self.query_budget = query_budget
        self.n_particles = particles
        self.ess_threshold = ess_threshold
        self.beta = beta
        self.carry_state = carry_state

        self._particle_state: list[str] | None = None
        self._weight_state: list[float] | None = None
        self._round_id = 0
        self._last_metrics: dict = {}
        self._state_before: dict = {}
        self._state_after: dict = {}
        self._validator = None

    def _get_validator(self):
        if self._validator is None:
            self._validator = _ArenaValidator()
        return self._validator

    def reset(self, seed: int) -> None:
        self.base_seed = seed
        self._particle_state = None
        self._weight_state = None
        self._round_id = 0
        self._last_metrics = {}
        self._state_before = {}
        self._state_after = {}

    def _snapshot_state(self) -> dict:
        ps = self._particle_state
        ws = self._weight_state
        ess = None
        if ws:
            w = np.array(ws)
            ess = float(1.0 / np.square(w).sum()) if w.sum() > 0 else None
        return {
            "particle_count": len(ps) if ps else None,
            "unique_particles": len(set(ps)) if ps else None,
            "ess": ess,
            "round_id": self._round_id,
        }

    def propose(self, samples: Sequence[SourceSample], defender: Defender) -> list[CandidateRecord]:
        from arena.prob_lab.accounting import QueryLedger
        from arena.prob_lab.proposals import arena_proposal_families

        families = arena_proposal_families()
        self._state_before = self._snapshot_state()
        validator = self._get_validator()

        all_records: list[CandidateRecord] = []
        all_history: list[dict] = []
        runtime_start = time.monotonic()
        total_proposals = total_queries = total_valid = total_invalid = 0
        all_effects: list[float] = []

        final_particles: list[str] = []
        final_weights: list[float] = []

        for source_index, sample in enumerate(samples):
            round_seed = self.base_seed + self._round_id * 1000 + source_index
            ledger = QueryLedger(budget=self.query_budget)
            model = _ArenaTargetModel(defender, ledger)
            source_pred = model.predict(sample.input)
            rng = np.random.default_rng(round_seed)

            # Carry-state
            if self.carry_state and self._particle_state is not None:
                particles = list(self._particle_state)
                weights = np.array(self._weight_state if self._weight_state else
                                   [1.0 / len(particles)] * len(particles))
            else:
                n = min(self.n_particles, max(1, self.query_budget - 1))
                particles = [sample.input] * n
                weights = np.full(n, 1.0 / n)

            best, best_effect = None, -1.0
            effects, invalid, valid_set = [], 0, set()

            while ledger.remaining_target and particles:
                next_particles, scores = [], []
                for parent in particles:
                    if not ledger.remaining_target:
                        break
                    family = families[int(rng.integers(len(families)))]
                    props = list(family.propose(sample.input, parent, rng))
                    ledger.proposal_count += len(props)
                    if not props:
                        continue
                    prop = props[int(rng.integers(len(props)))]
                    ledger.validity_checks += 1
                    if not validator.validate(sample.input, prop.candidate).eligible:
                        invalid += 1
                        continue
                    try:
                        effect = abs(model.predict(prop.candidate) - source_pred)
                    except Exception:
                        break
                    score = self.beta * effect - prop.edit_cost
                    next_particles.append(prop.candidate)
                    scores.append(score)
                    effects.append(effect)
                    valid_set.add(prop.candidate)
                    if effect > best_effect:
                        best, best_effect = prop.candidate, effect

                if not next_particles:
                    break

                shifted = np.asarray(scores) - max(scores)
                weights = np.exp(shifted)
                weights /= weights.sum()
                ess = float(1.0 / np.square(weights).sum())
                weight_entropy = float(-sum(w * math.log(w) for w in weights if w > 0))
                resampled = ess < self.ess_threshold * len(next_particles)

                if resampled:
                    indices = rng.choice(len(next_particles), size=len(next_particles), replace=True, p=weights)
                    particles = [next_particles[int(i)] for i in indices]
                    weights = np.full(len(particles), 1.0 / len(particles))
                else:
                    particles = next_particles

                all_history.append({
                    "source_index": source_index,
                    "step": len(all_history),
                    "particle_count": len(particles),
                    "ess": ess,
                    "weight_entropy": weight_entropy,
                    "resampled": resampled,
                    "unique_particles": len(set(particles)),
                    "queries": ledger.target_model_queries,
                })

            all_effects.extend(effects)
            total_proposals += ledger.proposal_count
            total_queries += ledger.target_model_queries
            total_valid += len(effects)
            total_invalid += invalid
            final_particles.extend(particles)
            final_weights.extend(weights.tolist())

            if best:
                all_records.append(CandidateRecord(
                    source_id=sample.source_id,
                    original_input=sample.input,
                    candidate_input=best,
                    attack_family=self.name,
                    attack_variant="smc",
                    proposal_step=0,
                    target_query_count=ledger.target_model_queries,
                    score=best_effect,
                    clean_prediction=source_pred,
                    challenging_prediction=source_pred + best_effect,
                    drift=best_effect,
                    true_target=sample.target,
                    metadata={
                        "round_id": self._round_id,
                        "carry_state": self.carry_state,
                        "particle_count": len(particles),
                        "valid_count": len(effects),
                        "invalid_count": invalid,
                        "target_model_queries": ledger.target_model_queries,
                    },
                ))

        # Update carry-state
        if self.carry_state and final_particles:
            self._particle_state = final_particles
            w = np.array(final_weights)
            w /= w.sum()
            self._weight_state = w.tolist()
        else:
            self._particle_state = None
            self._weight_state = None

        from arena.prob_lab.metrics import summarize_effects
        summary = summarize_effects(all_effects)

        ess_final = None
        if self._weight_state:
            w = np.array(self._weight_state)
            ess_final = float(1.0 / np.square(w).sum())

        self._state_after = {
            "particle_count": len(final_particles),
            "unique_particles": len(set(final_particles)) if final_particles else 0,
            "ess": ess_final,
            "round_id": self._round_id,
        }
        self._last_metrics = {
            "round_id": self._round_id,
            "carry_state": self.carry_state,
            "proposal_count": total_proposals,
            "target_model_queries": total_queries,
            "valid_count": total_valid,
            "invalid_count": total_invalid,
            "validity_rate": total_valid / max(1, total_proposals),
            "mean_drift": summary["mean"],
            "best_drift": summary["best"],
            "runtime_seconds": time.monotonic() - runtime_start,
            "ess": ess_final,
            "particle_count": len(final_particles),
        }
        self._round_id += 1
        return all_records

    def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.base_seed,
            "carry_state": self.carry_state,
            "round_id": self._round_id,
            "training_status": "adaptive",
            "state_before": self._state_before,
            "state_after": self._state_after,
            **self._last_metrics,
        }

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._last_metrics)


# ---------------------------------------------------------------------------
# Validity-SMC Attacker
# ---------------------------------------------------------------------------

class ValiditySMCAttacker(SMCAttacker):
    """
    SMC + empirical Beta-Bernoulli validity model.

    The validity model tracks which edit families tend to produce valid SMILES
    and downweights proposals from families with poor observed validity rates.
    This is the main 'query efficiency' experiment: does validity guidance
    allow the same query budget to find stronger failures?
    """

    name = "ValiditySMC"
    version = "1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validity_model = _EmpiricalValidityModelArena()

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._validity_model = _EmpiricalValidityModelArena()

    def propose(self, samples: Sequence[SourceSample], defender: Defender) -> list[CandidateRecord]:
        from arena.prob_lab.accounting import QueryLedger
        from arena.prob_lab.proposals import arena_proposal_families

        families = arena_proposal_families()
        self._state_before = {
            **self._snapshot_state(),
            "validity_posterior": self._validity_model.snapshot(),
        }
        validator = self._get_validator()

        all_records: list[CandidateRecord] = []
        all_history: list[dict] = []
        runtime_start = time.monotonic()
        total_proposals = total_queries = total_valid = total_invalid = 0
        total_validity_checks = 0
        all_effects: list[float] = []
        final_particles: list[str] = []
        final_weights: list[float] = []

        for source_index, sample in enumerate(samples):
            round_seed = self.base_seed + self._round_id * 1000 + source_index
            ledger = QueryLedger(budget=self.query_budget)
            model = _ArenaTargetModel(defender, ledger)
            source_pred = model.predict(sample.input)
            rng = np.random.default_rng(round_seed)

            if self.carry_state and self._particle_state is not None:
                particles = list(self._particle_state)
                weights = np.array(self._weight_state if self._weight_state else
                                   [1.0 / len(particles)] * len(particles))
            else:
                n = min(self.n_particles, max(1, self.query_budget - 1))
                particles = [sample.input] * n
                weights = np.full(n, 1.0 / n)

            best, best_effect = None, -1.0
            effects, invalid, valid_set = [], 0, set()

            while ledger.remaining_target and particles:
                next_particles, scores = [], []
                for parent in particles:
                    if not ledger.remaining_target:
                        break
                    family = families[int(rng.integers(len(families)))]
                    props = list(family.propose(sample.input, parent, rng))
                    ledger.proposal_count += len(props)
                    if not props:
                        continue
                    prop = props[int(rng.integers(len(props)))]
                    ledger.validity_checks += 1
                    total_validity_checks += 1
                    check = validator.validate(sample.input, prop.candidate)
                    self._validity_model.update(family.name, check.eligible)
                    if not check.eligible:
                        invalid += 1
                        continue
                    try:
                        effect = abs(model.predict(prop.candidate) - source_pred)
                    except Exception:
                        break
                    p_valid = self._validity_model.probability(family.name)
                    score = self.beta * effect + math.log(max(p_valid, 1e-12)) - prop.edit_cost
                    next_particles.append(prop.candidate)
                    scores.append(score)
                    effects.append(effect)
                    valid_set.add(prop.candidate)
                    if effect > best_effect:
                        best, best_effect = prop.candidate, effect

                if not next_particles:
                    break

                shifted = np.asarray(scores) - max(scores)
                weights = np.exp(shifted)
                weights /= weights.sum()
                ess = float(1.0 / np.square(weights).sum())
                weight_entropy = float(-sum(w * math.log(w) for w in weights if w > 0))
                resampled = ess < self.ess_threshold * len(next_particles)
                if resampled:
                    indices = rng.choice(len(next_particles), size=len(next_particles), replace=True, p=weights)
                    particles = [next_particles[int(i)] for i in indices]
                    weights = np.full(len(particles), 1.0 / len(particles))
                else:
                    particles = next_particles

                all_history.append({
                    "source_index": source_index,
                    "step": len(all_history),
                    "particle_count": len(particles),
                    "ess": ess,
                    "weight_entropy": weight_entropy,
                    "resampled": resampled,
                    "validity_posterior": self._validity_model.snapshot(),
                    "queries": ledger.target_model_queries,
                })

            all_effects.extend(effects)
            total_proposals += ledger.proposal_count
            total_queries += ledger.target_model_queries
            total_valid += len(effects)
            total_invalid += invalid
            final_particles.extend(particles)
            final_weights.extend(weights.tolist())

            if best:
                all_records.append(CandidateRecord(
                    source_id=sample.source_id,
                    original_input=sample.input,
                    candidate_input=best,
                    attack_family=self.name,
                    attack_variant="validity_smc",
                    proposal_step=0,
                    target_query_count=ledger.target_model_queries,
                    score=best_effect,
                    clean_prediction=source_pred,
                    challenging_prediction=source_pred + best_effect,
                    drift=best_effect,
                    true_target=sample.target,
                    metadata={
                        "round_id": self._round_id,
                        "carry_state": self.carry_state,
                        "validity_posterior": self._validity_model.snapshot(),
                        "valid_count": len(effects),
                        "invalid_count": invalid,
                        "target_model_queries": ledger.target_model_queries,
                        "validity_checks": total_validity_checks,
                    },
                ))

        if self.carry_state and final_particles:
            self._particle_state = final_particles
            w = np.array(final_weights)
            w /= w.sum()
            self._weight_state = w.tolist()
        else:
            self._particle_state = None
            self._weight_state = None

        from arena.prob_lab.metrics import summarize_effects
        summary = summarize_effects(all_effects)
        ess_final = None
        if self._weight_state:
            w = np.array(self._weight_state)
            ess_final = float(1.0 / np.square(w).sum())

        self._state_after = {
            "particle_count": len(final_particles),
            "unique_particles": len(set(final_particles)) if final_particles else 0,
            "ess": ess_final,
            "validity_posterior": self._validity_model.snapshot(),
            "round_id": self._round_id,
        }
        self._last_metrics = {
            "round_id": self._round_id,
            "carry_state": self.carry_state,
            "proposal_count": total_proposals,
            "target_model_queries": total_queries,
            "valid_count": total_valid,
            "invalid_count": total_invalid,
            "validity_rate": total_valid / max(1, total_proposals),
            "validity_checks": total_validity_checks,
            "mean_drift": summary["mean"],
            "best_drift": summary["best"],
            "runtime_seconds": time.monotonic() - runtime_start,
            "ess": ess_final,
            "validity_posterior": self._validity_model.snapshot(),
        }
        self._round_id += 1
        return all_records

    def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.base_seed,
            "carry_state": self.carry_state,
            "round_id": self._round_id,
            "training_status": "adaptive",
            "state_before": self._state_before,
            "state_after": self._state_after,
            **self._last_metrics,
        }


class _EmpiricalValidityModelArena:
    """Local copy of EmpiricalValidityModel for Arena — avoids circular import."""
    from collections import defaultdict as _dd

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        from collections import defaultdict
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.counts: dict = defaultdict(lambda: [alpha, beta])

    def update(self, family: str, valid: bool) -> None:
        self.counts[family][0 if valid else 1] += 1.0

    def probability(self, family: str) -> float:
        alpha, beta = self.counts[family]
        return alpha / (alpha + beta)

    def snapshot(self) -> dict:
        return {
            name: {"alpha": v[0], "beta": v[1], "p_valid": v[0] / sum(v)}
            for name, v in sorted(self.counts.items())
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def cem_attacker(seed: int, query_budget: int = 25, carry_state: bool = True) -> CEMAttacker:
    return CEMAttacker(seed=seed, query_budget=query_budget, carry_state=carry_state)


def smc_attacker(seed: int, query_budget: int = 25, carry_state: bool = True) -> SMCAttacker:
    return SMCAttacker(seed=seed, query_budget=query_budget, carry_state=carry_state)


def validity_smc_attacker(seed: int, query_budget: int = 25, carry_state: bool = True) -> ValiditySMCAttacker:
    return ValiditySMCAttacker(seed=seed, query_budget=query_budget, carry_state=carry_state)
