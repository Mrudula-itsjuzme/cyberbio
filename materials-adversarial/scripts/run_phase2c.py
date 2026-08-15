"""Phase 2C: multi-seed robustness confirmation.

Trains 5 independent seeds x 2 conditions (baseline, Phase 2B adversarial) under
the exact Phase 2B controls, then evaluates each on clean validation and on a
fresh, matched adversarial validation set.

Controls held fixed across every run:
  - Phase 1 TargetScaler, loaded not refit (mean 330.713124, std 88.046275)
  - same tokenizer / vocabulary
  - same scaffold split artifact
  - same architecture and optimizer settings
  - same adversarial training set (train_aug_phase2b.csv)
  - same attack budgets

The ONLY thing varying within a condition is the training seed. The only thing
varying between conditions is adversarial augmentation.

TEST SET IS NEVER TOUCHED. Model selection uses validation MAE only, and this
script never loads the test split.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.training.train import train  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [20260815, 20260816, 20260817, 20260818, 20260819]
PHASE1_SCALER = "results/models/transformer_regressor/scaler.json"
ADV_TRAIN = "data/processed/train_aug_phase2b.csv"
ROOT = Path("results/phase2c")

# Attack seed is derived per training seed so each seed sees a DIFFERENT fresh
# candidate set, but baseline and defended within a seed see the SAME one.
# Attacks are generated from sample PSMILES, not model gradients, so a shared
# RNG seed yields identical candidates across models -> matched/paired design.
ATTACK_SEED_BASE = 20270000


def model_dir(condition: str, seed: int) -> Path:
    return ROOT / f"{condition}_seed{seed}"


def train_one(condition: str, seed: int) -> None:
    out = model_dir(condition, seed)
    if (out / "metrics.json").exists():
        logger.info("SKIP %s seed=%d (already trained)", condition, seed)
        return

    logger.info("=" * 70)
    logger.info("TRAIN %s | seed=%d", condition, seed)
    train(
        augmented_train_path=ADV_TRAIN if condition == "defended" else None,
        out_dir=str(out),
        scaler_path=PHASE1_SCALER,
        write_back_config=False,
        seed=seed,
    )


def attack_one(condition: str, seed: int) -> None:
    out_file = f"phase2c_{condition}_seed{seed}.jsonl"
    if Path("data/attacks", out_file).exists():
        logger.info("SKIP attacks %s seed=%d (already generated)", condition, seed)
        return

    cmd = [
        sys.executable, "-m", "materials_adv.attacks.run_attacks",
        "--model-dir", str(model_dir(condition, seed)),
        "--out-file", out_file,
        "--attack-seed", str(ATTACK_SEED_BASE + seed % 1000),
        "--threshold", "52.02",
    ]
    env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
    logger.info("ATTACK %s | seed=%d", condition, seed)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr[-2000:])
        raise RuntimeError(f"attack run failed for {condition} seed={seed}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for condition in ("baseline", "defended"):
            train_one(condition, seed)
    for seed in SEEDS:
        for condition in ("baseline", "defended"):
            attack_one(condition, seed)

    manifest = {
        "seeds": SEEDS,
        "conditions": ["baseline", "defended"],
        "scaler": PHASE1_SCALER,
        "adv_train": ADV_TRAIN,
        "attack_seed_base": ATTACK_SEED_BASE,
        "note": "test set never loaded; model selection on validation MAE only",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Phase 2C runs complete.")


if __name__ == "__main__":
    main()
