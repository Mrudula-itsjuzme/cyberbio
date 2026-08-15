"""Baseline evaluation. PENDING(torch + checkpoint).

Reports MAE / RMSE / R-squared via evaluation.metrics.regression_metrics, which
is implemented and tested already -- only the model side is blocked.
"""

from __future__ import annotations

from ..utils.pending import PendingImplementation


def evaluate(*args, **kwargs):
    raise PendingImplementation(
        what="evaluate(): requires a trained checkpoint",
        blocked_on="torch+dataset",
        unblocks_when="Experiment 1 has produced a checkpoint",
    )
