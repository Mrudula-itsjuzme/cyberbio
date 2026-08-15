"""Artifact persistence.

JSONL is the record format. Parquet is deliberately not used: pyarrow is absent
from this environment, and JSONL stays diffable and appendable, which matters for
an attack log that grows incrementally and must record rejected candidates too.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> int:
    """Write dicts as JSONL. Returns the number of rows written."""
    p = Path(path)
    ensure_dir(p.parent)
    mode = "a" if append else "w"
    n = 0
    with p.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_no}") from exc


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        fh.write("\n")
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
