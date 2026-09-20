from __future__ import annotations

import csv, json, os, platform, subprocess, sys
from pathlib import Path
from typing import Any


class RunStorage:
    def __init__(self, root: str | Path, run_id: str):
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)

    @staticmethod
    def write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True, default=str)+"\n" for row in rows))

    def metadata(self) -> dict:
        def git(*args):
            return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
        return {"git_commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "python": sys.version, "platform": platform.platform()}

    def transition(self, state: str, round_id: int | None = None, detail: str | None = None) -> None:
        path = self.path / "state_transitions.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps({"state": state, "round": round_id, "detail": detail})+"\n")

    def round_dir(self, round_id: int) -> Path:
        path = self.path / f"round_{round_id:03d}"; path.mkdir(exist_ok=True); return path

