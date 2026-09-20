from __future__ import annotations

import math
import numpy as np

from probabilistic_search_lab.search_common import begin, finalize


class SMCSearch:
    def __init__(self, families, particles=8, ess_threshold=0.5, beta=1.0, validity_model=None):
        self.families = list(families); self.particles = particles
        self.ess_threshold = ess_threshold; self.beta = beta
        self.validity_model = validity_model

    def run(self, *, source_id, source, model, validator, budget, seed):
        ledger, target, started = begin(model, budget); rng = np.random.default_rng(seed)
        source_pred = target.predict(source)
        particles = [source] * min(self.particles, max(1, budget - 1)); weights = np.full(len(particles), 1 / len(particles))
        effects, valid, invalid, best, best_effect, best_family, history = [], set(), 0, None, -1.0, None, []
        while ledger.remaining_target and particles:
            next_particles, scores, family_names = [], [], []
            for parent in particles:
                if not ledger.remaining_target: break
                if self.validity_model is None:
                    family = self.families[int(rng.integers(len(self.families)))]
                else:
                    family_probabilities = np.asarray(
                        [self.validity_model.probability(item.name) for item in self.families], dtype=float
                    )
                    family_probabilities /= family_probabilities.sum()
                    family = self.families[int(rng.choice(len(self.families), p=family_probabilities))]
                proposals = list(family.propose(source, parent, rng)); ledger.proposal_count += len(proposals)
                if not proposals: continue
                proposal = proposals[int(rng.integers(len(proposals)))]; ledger.validity_checks += 1
                check = validator.validate(source, proposal.candidate)
                if self.validity_model is not None:
                    self.validity_model.update(family.name, check.eligible)
                if not check.eligible: invalid += 1; continue
                effect = abs(target.predict(proposal.candidate) - source_pred)
                validity_probability = self.validity_model.probability(family.name) if self.validity_model else 1.0
                score = self.beta * effect + math.log(max(validity_probability, 1e-12)) - proposal.edit_cost
                next_particles.append(proposal.candidate); scores.append(score); family_names.append(family.name)
                effects.append(effect); valid.add(proposal.candidate)
                if effect > best_effect: best, best_effect, best_family = proposal.candidate, effect, family.name
            if not next_particles: break
            shifted = np.asarray(scores) - max(scores); weights = np.exp(shifted); weights /= weights.sum()
            ess = float(1.0 / np.square(weights).sum()); entropy = float(-sum(w * math.log(w) for w in weights if w > 0))
            resampled = ess < self.ess_threshold * len(next_particles)
            if resampled:
                indices = rng.choice(len(next_particles), size=len(next_particles), replace=True, p=weights)
                particles = [next_particles[int(i)] for i in indices]; weights = np.full(len(particles), 1 / len(particles))
            else:
                particles = next_particles
            history.append({"step": len(history), "particle_count": len(particles), "ess": ess,
                            "weight_entropy": entropy, "resampled": resampled,
                            "unique_particles": len(set(particles)), "queries": ledger.target_model_queries})
        method = "validity_smc" if self.validity_model is not None else "smc"
        metadata = {"particles": self.particles, "ess_threshold": self.ess_threshold, "beta": self.beta}
        metadata["best_edit_type"] = best_family
        if self.validity_model is not None: metadata["validity_posterior"] = self.validity_model.snapshot()
        return finalize(method=method, seed=seed, source_id=source_id, budget=budget,
            ledger=ledger, started=started, effects=effects, valid_candidates=valid,
            invalid_count=invalid, best_candidate=best, trajectories={"smc": history}, metadata=metadata)
