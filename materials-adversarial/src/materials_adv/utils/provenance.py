"""Experiment provenance capture.

Saves a machine-readable JSON record alongside result artifacts so every run
is self-describing and reproducible.  Import and call ``capture()`` at the
start of any experiment that writes a result file.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_info() -> dict[str, str | bool]:
    """Capture git commit SHA and dirty-tree state."""
    info: dict[str, str | bool] = {"git_commit": "UNKNOWN", "git_dirty": "UNKNOWN"}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_commit"] = sha
    except Exception:
        pass
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_dirty"] = bool(status)
    except Exception:
        pass
    return info


def _package_versions() -> dict[str, str]:
    """Capture versions of key packages."""
    pkgs = {}
    for pkg in ["torch", "numpy", "pandas", "rdkit", "scipy", "sklearn"]:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            pkgs[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkgs[pkg] = "not_installed"
    # rdkit special case
    try:
        from rdkit import __version__ as rdkit_ver
        pkgs["rdkit"] = rdkit_ver
    except Exception:
        pass
    return pkgs


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's contents."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "UNREADABLE"


def capture(
    *,
    seed: int | None,
    config: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    split_manifest: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect all provenance information for the current run.

    Parameters
    ----------
    seed:
        The random seed used for this run.
    config:
        A snapshot of the experiment configuration (plain-Python dict).
    dataset_path:
        Path to the processed dataset file for hashing.
    split_manifest:
        Path to the split JSON for hashing.
    checkpoint_path:
        Path to the model checkpoint for hashing.
    extra:
        Any additional key/value pairs to store.

    Returns
    -------
    dict
        The full provenance record.
    """
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_unix": time.time(),
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": "cuda" if _has_cuda() else "cpu",
        "packages": _package_versions(),
        "launch_command": " ".join(sys.argv),
        "working_directory": os.getcwd(),
    }

    record.update(_git_info())

    if config is not None:
        record["config_snapshot"] = config

    if dataset_path is not None:
        record["dataset_path"] = str(dataset_path)
        record["dataset_sha256"] = file_sha256(dataset_path)

    if split_manifest is not None:
        record["split_manifest_path"] = str(split_manifest)
        record["split_manifest_sha256"] = file_sha256(split_manifest)

    if checkpoint_path is not None:
        record["checkpoint_path"] = str(checkpoint_path)
        record["checkpoint_sha256"] = file_sha256(checkpoint_path)

    if extra:
        record["extra"] = extra

    return record


def save(record: dict[str, Any], output_path: str | Path) -> Path:
    """Save provenance record as JSON beside a result artifact.

    Parameters
    ----------
    record:
        Provenance dict returned by :func:`capture`.
    output_path:
        Path for the provenance JSON (e.g. ``results/run/provenance.json``).

    Returns
    -------
    Path
        The path where the file was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return output_path


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
