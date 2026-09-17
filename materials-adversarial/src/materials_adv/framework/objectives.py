"""Attack objectives.

An objective maps a PAIR of already-computed predictions to an attacker score. It
must not be able to reach the model itself: the forensic audit found that the
previous ``evaluate(predictor, source, candidate)`` contract spent **two** model
calls for every nominal query budget slot (candidate *and* source), which is one of
the reasons Framework V2's query accounting was invalid
(docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 12).

The budgeted scoring path is
:class:`materials_adv.framework.accounting.AttackEvaluator`, which computes the
source prediction once and each candidate prediction exactly once, then calls
:meth:`AttackObjective.score`.
"""

from __future__ import annotations

import abc

from materials_adv.framework.interfaces import Candidate, Predictor


class AttackObjective(abc.ABC):
    @abc.abstractmethod
    def score(self, source_prediction: float, candidate_prediction: float) -> float:
        """Attacker score for one candidate. Higher is better for the attacker.

        Both arguments are already-computed predictions in the model's own unit
        (eV for the polymer models). No model access is possible from here.
        """
        raise NotImplementedError

    def evaluate(self, predictor: Predictor, source: Candidate, candidate: Candidate) -> float:
        """Legacy convenience wrapper.

        NOT used by the budgeted evaluator: it would spend two model calls per
        candidate, so a search that uses it cannot honour a query budget.
        """
        return self.score(predictor.predict(source), predictor.predict(candidate))


class UntargetedDrift(AttackObjective):
    """maximise |f(x_adv) - f(x_src)|"""

    def score(self, source_prediction: float, candidate_prediction: float) -> float:
        return abs(candidate_prediction - source_prediction)


class TargetIncrease(AttackObjective):
    """maximise f(x_adv) - f(x_src)"""

    def score(self, source_prediction: float, candidate_prediction: float) -> float:
        return candidate_prediction - source_prediction


class TargetDecrease(AttackObjective):
    """maximise f(x_src) - f(x_adv)"""

    def score(self, source_prediction: float, candidate_prediction: float) -> float:
        return source_prediction - candidate_prediction


class TargetValue(AttackObjective):
    """Steer the prediction towards a chosen value."""

    def __init__(self, target_value: float):
        self.target_value = target_value

    def score(self, source_prediction: float, candidate_prediction: float) -> float:
        return -abs(candidate_prediction - self.target_value)
