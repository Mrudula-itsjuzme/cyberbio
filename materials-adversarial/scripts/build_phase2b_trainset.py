"""Phase 2B: build the deduplicated controlled adversarial training set.

Reuses the EXACT Phase 2A augmentation artifact (data/processed/train_aug.csv)
rather than regenerating it. Regenerating would re-run the attack RNG and change
which adversarial strings exist, introducing a second difference between 2A and
2B and defeating the point of the ablation.

Two duplicate classes are removed, and reported separately:
  1. adversarial-vs-adversarial: the same adversarial PSMILES generated twice
  2. adversarial-vs-clean:       an "adversarial" string identical to a clean
                                 training string. These are pure sample-weight
                                 inflation -- they add no new information and
                                 silently up-weight some clean polymers.

Phase 2A results are NOT modified. This writes new files only.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.utils.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BALANCE_SEED = 20260815


def build(balanced: bool = False) -> dict:
    data_cfg = load_config("configs/dataset.yaml")
    rep_col = data_cfg["representation_column"]
    proc_dir = Path(data_cfg["processed_dir"])

    aug = pd.read_csv(proc_dir / "train_aug.csv")
    clean = aug[~aug["is_adv"]].copy()
    adv = aug[aug["is_adv"]].copy()

    n_clean = len(clean)
    n_adv_raw = len(adv)
    raw_counts = adv["attack_type"].value_counts().to_dict()

    # 1. adversarial-vs-adversarial duplicates
    adv_dedup = adv.drop_duplicates(subset=[rep_col], keep="first")
    n_dup_adv = n_adv_raw - len(adv_dedup)

    # 2. adversarial strings identical to a clean training string
    clean_set = set(clean[rep_col])
    collides = adv_dedup[rep_col].isin(clean_set)
    n_dup_vs_clean = int(collides.sum())
    adv_dedup = adv_dedup[~collides]

    n_adv_final = len(adv_dedup)
    dedup_counts = adv_dedup["attack_type"].value_counts().to_dict()

    balance_note = None
    if balanced:
        smallest = min(dedup_counts.values())
        # Sample per family by index, then reselect: groupby.apply would consume
        # the grouping column and drop 'attack_type' from the result.
        keep_idx = []
        for _, group in adv_dedup.groupby("attack_type"):
            keep_idx.extend(group.sample(n=smallest, random_state=BALANCE_SEED).index)
        adv_dedup = adv_dedup.loc[keep_idx].reset_index(drop=True)
        balance_note = (
            f"Down-sampled every family to {smallest} "
            f"(rearrangement is the binding constraint), "
            f"discarding {n_adv_final - len(adv_dedup)} examples "
            f"({100 * (n_adv_final - len(adv_dedup)) / n_adv_final:.1f}%)."
        )
        logger.warning("BALANCED VARIANT: %s", balance_note)

    combined = pd.concat([clean, adv_dedup], ignore_index=True)

    suffix = "_balanced" if balanced else ""
    out_path = proc_dir / f"train_aug_phase2b{suffix}.csv"
    combined.to_csv(out_path, index=False)

    report = {
        "variant": "balanced" if balanced else "natural_dedup",
        "clean_training_samples": n_clean,
        "adversarial_before_dedup": n_adv_raw,
        "duplicates_adv_vs_adv_removed": n_dup_adv,
        "duplicates_adv_vs_clean_removed": n_dup_vs_clean,
        "total_duplicates_removed": n_dup_adv + n_dup_vs_clean,
        "adversarial_after_dedup": int(len(adv_dedup)),
        "total_training_rows": len(combined),
        "attack_family_counts_raw": raw_counts,
        "attack_family_counts_after_dedup": dedup_counts,
        "attack_family_counts_final": adv_dedup["attack_type"].value_counts().to_dict(),
        "balance_note": balance_note,
        "output": str(out_path),
    }

    logger.info("Wrote %s (%d rows)", out_path, len(combined))
    return report


if __name__ == "__main__":
    reports = {"natural_dedup": build(balanced=False), "balanced": build(balanced=True)}
    out = Path("results/phase2b_trainset_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))
