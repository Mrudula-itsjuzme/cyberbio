"""Attack registry, so attacks are pluggable.

Adding an attack means writing one file and decorating the class. The evaluator
resolves attacks by name from config and never needs to change.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import numpy as np

from .base import BaseAttack

_REGISTRY: dict[str, type[BaseAttack]] = {}

T = TypeVar("T", bound=type[BaseAttack])


def register_attack(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Attack name {name!r} is already registered to {_REGISTRY[name]!r}")
        cls.name = name  # type: ignore[misc]
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_attacks() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_attack_class(name: str) -> type[BaseAttack]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown attack {name!r}. Available: {available_attacks()}")
    return _REGISTRY[name]


def build_attack(name: str, rng: np.random.Generator, **params: Any) -> BaseAttack:
    return get_attack_class(name)(rng, **params)
