"""Phase 2B: controlled defense ablation.

The ONLY intended difference from the Phase 1 baseline is adversarial
augmentation. Normalization statistics are held fixed by loading the exact
Phase 1 TargetScaler instead of refitting it -- the Phase 2A confounder.

Everything else is inherited unchanged from configs/model.yaml: architecture,
tokenizer, split, optimizer, learning rate, weight decay, epoch budget, seed and
evaluation procedure.

configs/model.yaml is NOT written back (write_back_config=False), because the
Phase 2A run overwrote its `baseline_metrics` with the defended model's numbers.
Each run's metrics are saved next to its own checkpoint instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.training.train import train  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PHASE1_SCALER = "results/models/transformer_regressor/scaler.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["natural", "balanced"],
        default="natural",
        help="natural = deduplicated mixture (primary); balanced = equal families (diagnostic)",
    )
    args = parser.parse_args()

    if args.variant == "balanced":
        train_path = "data/processed/train_aug_phase2b_balanced.csv"
        out_dir = "results/models/transformer_defended_phase2b_balanced"
    else:
        train_path = "data/processed/train_aug_phase2b.csv"
        out_dir = "results/models/transformer_defended_phase2b"

    logger.info("PHASE 2B (%s): training with FIXED Phase 1 scaler", args.variant)
    logger.info("  train data : %s", train_path)
    logger.info("  scaler     : %s (loaded, not refit)", PHASE1_SCALER)
    logger.info("  out dir    : %s", out_dir)

    train(
        augmented_train_path=train_path,
        out_dir=out_dir,
        scaler_path=PHASE1_SCALER,
        write_back_config=False,
    )
    logger.info("Phase 2B (%s) training complete.", args.variant)


if __name__ == "__main__":
    main()
