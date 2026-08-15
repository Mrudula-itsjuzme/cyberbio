"""Deterministic seeding.

The sibling project (bio-cyber-adversarial) seeded data generation but not its
training loop, so its checkpoint was not reproducible while its README claimed
"fully deterministic". Every stochastic entry point here takes an explicit seed.

Design rule: prefer an explicit `numpy.random.Generator` passed as an argument
over mutating global state. Attacks take `rng` in their constructor for exactly
this reason -- global seeding is a fallback for third-party code we don't control.
"""

from __future__ import annotations

import os
import random

import numpy as np

from .optional import has_torch

DEFAULT_SEED = 20260815


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent Generator. Preferred over global seeding."""
    return np.random.default_rng(seed)


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> dict[str, object]:
    """Seed all global RNGs we can reach. Returns a record of what was seeded.

    The returned dict is meant to be written into an experiment's metadata so a
    run's determinism guarantees are auditable after the fact.
    """
    record: dict[str, object] = {"seed": seed, "python_random": True, "numpy": True}

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    record["pythonhashseed"] = str(seed)

    # torch is optional and absent in the current environment; seeding it is
    # best-effort and reported honestly rather than assumed.
    if has_torch():
        import torch  # noqa: PLC0415  -- deliberately lazy, see utils/optional.py

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        record["torch"] = True
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            record["torch_deterministic"] = True
    else:
        record["torch"] = False

    return record
