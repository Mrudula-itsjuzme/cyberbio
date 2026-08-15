"""Attack pipeline orchestration.

    Original PSMILES -> AttackGenerator -> candidates -> validity filter
      -> target model -> prediction drift -> AttackRecord

Every candidate produces a record, including rejected and unscored ones. The
model is accessed through a minimal `predict` protocol, so attacks stay black-box
and a gradient-based attack can be added later without touching this file.
"""

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
    """Minimal target-model interface.

    Batch-first: a per-sample signature would make batched inference impossible
    to retrofit. `target_units` forces units to be explicit rather than implied,
    given the K-vs-Celsius ambiguity flagged in the dataset audit.
    """

    def predict(self, psmiles: Sequence[str]) -> np.ndarray: ...

    @property
    def target_units(self) -> str: ...


class ConstantPredictor:
    """Test double returning a fixed value.

    Lets the entire attack + validation + metrics pipeline run end-to-end with no
    torch installed, so the plumbing is verifiable before a real model exists.
    Never use for research results.
    """

    def __init__(self, value: float = 300.0, units: str = "K") -> None:
        self.value = value
        self._units = units

    def predict(self, psmiles: Sequence[str]) -> np.ndarray:
        return np.full(len(psmiles), self.value, dtype=float)

    @property
    def target_units(self) -> str:
        return self._units


class AttackGenerator:
    """Runs one or more attacks over samples and produces AttackRecords."""

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
            outcome.adversarial_psmiles, check_plausibility=self.check_plausibility
        )
        return AttackRecord(
            attack_id=make_attack_id(sample_id, outcome.attack_type, self.seed, ordinal),
            sample_id=sample_id,
            original_psmiles=outcome.original_psmiles,
            adversarial_psmiles=outcome.adversarial_psmiles,
            attack_type=outcome.attack_type,
            number_of_changes=outcome.number_of_changes,
            validity_status=result.status.value,
            original_prediction=original_pred,
            adversarial_prediction=adversarial_pred,
            prediction_drift=compute_drift(original_pred, adversarial_pred),
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
                preds = self.predictor.predict([o.adversarial_psmiles for o in outcomes])
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
        """Run over (sample_id, psmiles) pairs."""
        records: list[AttackRecord] = []
        for sample_id, psmiles in samples:
            records.extend(self.run_sample(sample_id, psmiles, n_variants=n_variants))
        return records
