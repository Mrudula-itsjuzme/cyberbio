"""Model registry, mirroring the attack registry so models are pluggable too.

Import-free indirection: registering a name does not import torch, so this module
stays usable in the current environment.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

_REGISTRY: dict[str, Callable[..., Any]] = {}

F = TypeVar("F", bound=Callable[..., Any])


def register_model(name: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"Model name {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_model(name: str, **kwargs: Any) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")
    return _REGISTRY[name](**kwargs)
