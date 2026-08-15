"""Lazy access to optional heavy dependencies (torch, rdkit).

Nothing in this package may import torch or rdkit at module level. Doing so would
make the whole scaffold unimportable in an environment where they are absent --
which is the current environment. All access goes through `require()`.

`has_*()` predicates drive graceful degradation: a missing dependency means a
check is *skipped and reported as skipped*, never that it *failed*.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

# extra name -> pip install hint
_INSTALL_HINTS: dict[str, str] = {
    "chem": 'pip install -e ".[chem]"',
    "model": (
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        "   # CPU wheel: this machine has no GPU"
    ),
}


class MissingDependency(ImportError):
    """A required optional dependency is not installed."""


def is_available(module_name: str) -> bool:
    """True if `module_name` can be imported, without actually importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def require(module_name: str, extra: str) -> ModuleType:
    """Import `module_name`, or raise with the exact command that installs it."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = _INSTALL_HINTS.get(extra, f'pip install -e ".[{extra}]"')
        raise MissingDependency(
            f"'{module_name}' is required for this operation but is not installed.\n"
            f"  Install with: {hint}"
        ) from exc


def has_rdkit() -> bool:
    return is_available("rdkit")


def has_torch() -> bool:
    return is_available("torch")
