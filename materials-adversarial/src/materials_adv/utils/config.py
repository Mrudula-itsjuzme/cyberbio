"""YAML config loading.

Configs rather than hardcoded values, per the engineering requirements. Plain
PyYAML + dataclasses is deliberate: a heavier framework's config resolution
obscures what actually ran, which is bad for reproducibility claims.

A `null` value in a config is meaningful -- it marks a decision that is PENDING
because the dataset has not been audited yet. `require_resolved()` turns reading
such a value into a loud failure rather than a silent `None` propagating into a
model constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .pending import PendingImplementation

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by bare name ('model') or explicit path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return loaded if loaded is not None else {}


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value by dotted path, e.g. 'transformer.n_layers'."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require_resolved(config: dict[str, Any], dotted_key: str, *, unblocks_when: str) -> Any:
    """Fetch a value that must not still be PENDING.

    Raises PendingImplementation if the key is missing or null, so an unmade
    research decision cannot leak into a run as an implicit None.
    """
    value = get(config, dotted_key, default=None)
    if value is None:
        raise PendingImplementation(
            what=f"config key '{dotted_key}' is null (decision not yet made)",
            blocked_on="dataset-audit",
            unblocks_when=unblocks_when,
        )
    return value
