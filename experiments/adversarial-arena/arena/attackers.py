from __future__ import annotations

from typing import Sequence
import numpy as np

from .contracts import Attacker, CandidateRecord, Defender, SourceSample


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
        from materials_adv.data.tokenizer import tokenize

        attack = self.factory(self.rng, defender)
        records: list[CandidateRecord] = []
        for sample in samples:
            outcomes = attack.generate(tokenize(sample.input), n_variants=self.proposal_budget)
            for step, outcome in enumerate(outcomes):
                params = dict(outcome.params)
                queries = int(params.get("search_candidate_model_queries", 0))
                records.append(CandidateRecord(
                    source_id=sample.source_id, original_input=sample.input,
                    candidate_input=outcome.adversarial_representation,
                    attack_family=self.name, attack_variant=outcome.attack_type,
                    proposal_step=step, target_query_count=queries,
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

