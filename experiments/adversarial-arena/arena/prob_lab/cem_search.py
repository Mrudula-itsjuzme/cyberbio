"""
CEMSearch — extracted unchanged from experiments/probabilistic-search-lab (commit 4b8e172).
Import path adjusted from probabilistic_search_lab.* to arena.prob_lab.*.
"""
from __future__ import annotations

import numpy as np

from arena.prob_lab.metrics import categorical_entropy
from arena.prob_lab.search_common import begin, finalize


class CEMSearch:
    def __init__(self, families, batch_size=8, elite_fraction=0.25, smoothing=0.7, entropy_floor=0.05):
        self.families = list(families)
        self.batch_size = batch_size
        self.elite_fraction = elite_fraction
        self.smoothing = smoothing
        self.entropy_floor = entropy_floor

    def run(self, *, source_id, source, model, validator, budget, seed):
        ledger, target, started = begin(model, budget)
        rng = np.random.default_rng(seed)
        source_pred = target.predict(source)
        k = len(self.families)
        probs = np.full(k, 1.0 / k)
        effects, valid, invalid = [], set(), 0
        best, best_effect, history = None, -1.0, []

        while ledger.remaining_target:
            evaluated = []
            attempts = 0
            while len(evaluated) < min(self.batch_size, ledger.remaining_target) and attempts < self.batch_size * 20:
                attempts += 1
                family_index = int(rng.choice(k, p=probs))
                family = self.families[family_index]
                proposals = list(family.propose(source, source, rng))
                ledger.proposal_count += len(proposals)
                if not proposals:
                    continue
                proposal = proposals[int(rng.integers(len(proposals)))]
                ledger.validity_checks += 1
                if not validator.validate(source, proposal.candidate).eligible:
                    invalid += 1
                    continue
                effect = abs(target.predict(proposal.candidate) - source_pred)
                effects.append(effect)
                valid.add(proposal.candidate)
                evaluated.append((effect, family_index))
                if effect > best_effect:
                    best, best_effect = proposal.candidate, effect

            if not evaluated:
                break

            elite_n = max(1, int(np.ceil(len(evaluated) * self.elite_fraction)))
            elites = sorted(evaluated, reverse=True)[:elite_n]
            fitted = np.bincount([index for _, index in elites], minlength=k).astype(float)
            fitted = fitted / fitted.sum()
            fitted = (1.0 - self.entropy_floor) * fitted + self.entropy_floor / k
            probs = self.smoothing * fitted + (1.0 - self.smoothing) * probs
            probs /= probs.sum()
            history.append({
                "iteration": len(history),
                "probabilities": probs.tolist(),
                "entropy": categorical_entropy(probs.tolist()),
                "elite_effects": [effect for effect, _ in elites],
                "queries": ledger.target_model_queries,
            })

        return finalize(
            method="cem", seed=seed, source_id=source_id, budget=budget,
            ledger=ledger, started=started, effects=effects, valid_candidates=valid,
            invalid_count=invalid, best_candidate=best,
            trajectories={"cem": history},
            metadata={
                "elite_fraction": self.elite_fraction,
                "smoothing": self.smoothing,
                "entropy_floor": self.entropy_floor,
            },
        )
