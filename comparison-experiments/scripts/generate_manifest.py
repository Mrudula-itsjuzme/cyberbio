import json
import hashlib
import subprocess
from pathlib import Path

def get_git_commit(repo_path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path).decode("utf-8").strip()
    except Exception:
        return "unknown"

def generate_manifest(out_dir: Path, args, model_checkpoint: Path, source_ids_path: Path):
    manifest = {
        "args": vars(args),
        "environment": "materials-adversarial/.venv",
        "commit_sha": get_git_commit(out_dir.parent.parent),
        "model_hash": hashlib.md5(model_checkpoint.read_bytes()).hexdigest() if model_checkpoint.exists() else "missing",
        "source_bank_hash": hashlib.md5(source_ids_path.read_bytes()).hexdigest() if source_ids_path.exists() else "missing",
    }
    with open(out_dir / "reproducibility_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
