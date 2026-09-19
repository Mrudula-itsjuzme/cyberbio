import os
import json
import hashlib
import subprocess

def get_hash(f):
    if not os.path.exists(f): return "missing"
    return hashlib.sha256(open(f, "rb").read()).hexdigest()

try:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_sha = "unknown"

repro = {
    "git_SHA": git_sha,
    "dataset_SHA256": get_hash("data/v3/dataset.csv"),
    "checkpoint_SHA256": get_hash("results/v3/models/cnn_distance/model.pt"),
    "frozen_bank_SHA256": get_hash("results/v3/frozen_attack_banks/manifest.json")
}
with open("../docs/reproducibility_manifest.json", "w") as f:
    json.dump(repro, f, indent=4)
print("Repro fixed.")
