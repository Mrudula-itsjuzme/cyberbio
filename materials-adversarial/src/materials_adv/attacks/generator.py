from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ..data.tokenizer import tokenize
from ..evaluation.records import AttackRecord, compute_drift, make_attack_id
from ..validation.pipeline import validate
from .base import AttackOutcome, BaseAttack


@runtime_checkable
class PredictorProtocol(Protocol):
    def predict(self, psmiles: Sequence[str]) -> np.ndarray: ...

    @property
    def target_units(self) -> str: ...


class ConstantPredictor:
    def __init__(self, value: float = 300.0, units: str = "K") -> None:
        self.value = value
        self._units = units

    def predict(self, psmiles: Sequence[str]) -> np.ndarray:
        return np.full(len(psmiles), self.value, dtype=float)

    @property
    def target_units(self) -> str:
        return self._units


class AttackGenerator:
    def __init__(
        self,
        attacks: Sequence[BaseAttack],
        predictor: PredictorProtocol | None = None,
        *,
        seed: int | None = None,
        check_plausibility: bool = True,
    ) -> None:
        if not attacks:
            raise ValueError("AttackGenerator requires at least one attack")
        self.attacks = list(attacks)
        self.predictor = predictor
        self.seed = seed
        self.check_plausibility = check_plausibility

    def _record(
        self,
        sample_id: str,
        outcome: AttackOutcome,
        ordinal: int,
        original_pred: float | None,
        adversarial_pred: float | None,
        attack: BaseAttack,
    ) -> AttackRecord:
        result = validate(
            outcome.adversarial_representation, check_plausibility=self.check_plausibility
        )
        
        drift = compute_drift(original_pred, adversarial_pred)
        abs_drift = abs(drift) if drift is not None else None
        
        # Determine success (e.g. drift > threshold). For now we don't have a fixed threshold here,
        # but usually it's set later or based on some MAE. Let's default to False unless specified.
        # But wait, we can store False.
        success = False 
        
        plaus_status = "plausible" if result.status.value == "valid" else ("implausible" if result.status.value == "implausible" else "unchecked")
        val_status = "valid" if result.status.value != "invalid_representation" else "invalid_representation"

        # attack budget
        budget = attack.metadata().get("attack_budget", 1)

        return AttackRecord(
            attack_id=make_attack_id(sample_id, outcome.attack_type, self.seed, ordinal),
            sample_id=sample_id,
            original_representation=outcome.original_representation,
            adversarial_representation=outcome.adversarial_representation,
            attack_type=outcome.attack_type,
            attack_budget=budget,
            edited_positions=outcome.edit_positions,
            number_of_changes=outcome.number_of_changes,
            validity_status=val_status,
            plausibility_status=plaus_status,
            original_prediction=original_pred,
            adversarial_prediction=adversarial_pred,
            signed_prediction_drift=drift,
            absolute_prediction_drift=abs_drift,
            attack_success=success,
            rejection_reasons=result.rejection_reasons,
            plausibility_flags=result.plausibility_flags,
            checks_skipped=result.checks_skipped,
            seed=self.seed,
            attack_params=attack.metadata(),
        )

    def run_sample(
        self, sample_id: str, psmiles: str, n_variants: int = 1
    ) -> list[AttackRecord]:
        tokens = tokenize(psmiles)

        original_pred: float | None = None
        if self.predictor is not None:
            original_pred = float(self.predictor.predict([psmiles])[0])

        records: list[AttackRecord] = []
        for attack in self.attacks:
            outcomes = attack.generate(tokens, n_variants=n_variants)
            if not outcomes:
                continue

            adversarial_preds: list[float | None] = [None] * len(outcomes)
            if self.predictor is not None:
                preds = self.predictor.predict([o.adversarial_representation for o in outcomes])
                adversarial_preds = [float(p) for p in preds]

            for ordinal, (outcome, adv_pred) in enumerate(zip(outcomes, adversarial_preds)):
                records.append(
                    self._record(
                        sample_id, outcome, ordinal, original_pred, adv_pred, attack
                    )
                )
        return records

    def run(
        self,
        samples: Sequence[tuple[str, str]],
        n_variants: int = 1,
    ) -> list[AttackRecord]:
        records: list[AttackRecord] = []
        for sample_id, psmiles in samples:
            records.extend(self.run_sample(sample_id, psmiles, n_variants=n_variants))
        return records

