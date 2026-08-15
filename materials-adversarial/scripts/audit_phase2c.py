"""Phase 2C: aggregate multi-seed results.

Reports per-seed and across-seed mean +/- SD for clean validation metrics and
per-family adversarial drift.

On statistics: with n=5 seeds we report descriptive statistics and a paired
per-candidate comparison. A Wilcoxon signed-rank test over matched candidates is
included ONLY where its assumptions are met (paired observations, same candidate
scored by both models). It is NOT applied across seeds -- 5 points is too few to
support a distributional claim, and the seeds are not independent samples of any
population of interest. No significance claim is made about the clean metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.data.scaler import TargetScaler  # noqa: E402
from materials_adv.data.tokenizer import tokenize  # noqa: E402
from materials_adv.models.transformer import TransformerRegressorModel  # noqa: E402
from materials_adv.utils.config import load_config  # noqa: E402

SEEDS = [20260815, 20260816, 20260817, 20260818, 20260819]
FAMILIES = ["substitution", "insertion", "deletion", "rearrangement"]
THRESHOLD = 52.02
ROOT = Path("results/phase2c")


def encode(psmiles, vocab, max_len):
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    ids = [char2idx.get(t, 0) for t in tokenize(psmiles)]
    mask = [False] * len(ids)
    while len(ids) < max_len:
        ids.append(0)
        mask.append(True)
    return ids[:max_len], mask[:max_len]


def clean_val_metrics(model_dir: Path, val_df, rep_col, tgt_col, vocab, cfg) -> dict:
    arch = cfg["architecture"]
    model = TransformerRegressorModel(
        vocab_size=len(vocab), d_model=arch["d_model"], n_layers=arch["n_layers"],
        n_heads=arch["n_heads"], dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"], max_seq_len=arch["max_seq_len"], pooling=arch["pooling"],
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    scaler = TargetScaler.load(model_dir / "scaler.json")

    ids, masks = [], []
    for s in val_df[rep_col]:
        i, m = encode(s, vocab, arch["max_seq_len"])
        ids.append(i)
        masks.append(m)
    with torch.no_grad():
        out = model(torch.tensor(ids), padding_mask=torch.tensor(masks))
    pred = np.asarray(scaler.inverse_transform(out.numpy()))
    y = val_df[tgt_col].to_numpy(float)
    err = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1 - float(np.sum(err**2)) / ss_tot),
    }


def load_records(condition: str, seed: int) -> list[dict]:
    path = Path("data/attacks") / f"phase2c_{condition}_seed{seed}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def family_stats(records: list[dict]) -> dict[str, dict]:
    out = {}
    for fam in FAMILIES:
        d = [
            abs(r["prediction_drift"])
            for r in records
            if r["attack_type"] == fam
            and r["validity_status"] == "valid"
            and r["prediction_drift"] is not None
        ]
        if not d:
            continue
        a = np.asarray(d)
        out[fam] = {
            "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "frac_above": float((a > THRESHOLD).mean()),
        }
    return out


def msd(values: list[float]) -> str:
    a = np.asarray(values, dtype=float)
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"


def main() -> None:
    data_cfg, cfg = load_config("configs/dataset.yaml"), load_config("configs/model.yaml")
    rep_col, tgt_col = data_cfg["representation_column"], data_cfg["target_column"]
    proc = Path(data_cfg["processed_dir"])
    df = pd.read_csv(proc / "processed.csv")
    splits = json.loads((proc / "splits.json").read_text())
    vocab = json.loads((proc / "vocab.json").read_text())
    val_df = df.iloc[splits["val"]]

    results = {"seeds": SEEDS, "threshold_K": THRESHOLD, "per_seed": {}}

    print("=" * 84)
    print(f"{'PHASE 2C: MULTI-SEED ROBUSTNESS CONFIRMATION':^84}")
    print("=" * 84)
    print(f"\nValidation n={len(val_df)}.  TEST SET NOT USED.\n")

    # --- Clean validation ---
    print("--- CLEAN VALIDATION (per seed) ---")
    print(f"{'seed':>10} {'baseline MAE':>14} {'defended MAE':>14} {'delta':>9} "
          f"{'base R2':>9} {'def R2':>9}")
    clean = {"baseline": [], "defended": []}
    for seed in SEEDS:
        row = {}
        for cond in ("baseline", "defended"):
            row[cond] = clean_val_metrics(
                ROOT / f"{cond}_seed{seed}", val_df, rep_col, tgt_col, vocab, cfg
            )
            clean[cond].append(row[cond])
        results["per_seed"][str(seed)] = {"clean": row}
        print(f"{seed:>10} {row['baseline']['mae']:>14.2f} {row['defended']['mae']:>14.2f} "
              f"{row['defended']['mae'] - row['baseline']['mae']:>+9.2f} "
              f"{row['baseline']['r2']:>9.2f} {row['defended']['r2']:>9.2f}")

    print("\n--- CLEAN VALIDATION (mean ± SD across 5 seeds) ---")
    for metric in ("mae", "rmse", "r2"):
        b = [c[metric] for c in clean["baseline"]]
        d = [c[metric] for c in clean["defended"]]
        print(f"  {metric.upper():>5}   baseline {msd(b):>16}    defended {msd(d):>16}")
        results.setdefault("clean_summary", {})[metric] = {
            "baseline_mean": float(np.mean(b)), "baseline_sd": float(np.std(b, ddof=1)),
            "defended_mean": float(np.mean(d)), "defended_sd": float(np.std(d, ddof=1)),
        }
    mae_diff = [d["mae"] - b["mae"] for b, d in zip(clean["baseline"], clean["defended"])]
    results["clean_summary"]["mae_difference"] = {
        "per_seed": mae_diff,
        "mean": float(np.mean(mae_diff)),
        "sd": float(np.std(mae_diff, ddof=1)),
    }
    print(f"\n  Clean MAE difference (defended - baseline): {msd(mae_diff)} K")
    print(f"    per-seed: {[round(x, 2) for x in mae_diff]}")

    # --- Robustness ---
    print("\n--- ADVERSARIAL VALIDATION: per-family mean drift, per seed (base -> def) ---")
    per_seed_fam: dict[str, dict[str, list]] = {f: {"baseline": [], "defended": []} for f in FAMILIES}
    pooled = {"baseline": [], "defended": []}
    paired_diffs: list[float] = []
    integrity_ok = True

    hdr = f"{'seed':>10} " + " ".join(f"{f[:5]:>16}" for f in FAMILIES)
    print(hdr)
    for seed in SEEDS:
        rb, rd = load_records("baseline", seed), load_records("defended", seed)
        same = [x["adversarial_psmiles"] for x in rb] == [x["adversarial_psmiles"] for x in rd]
        integrity_ok &= same

        sb, sd_ = family_stats(rb), family_stats(rd)
        cells = []
        for fam in FAMILIES:
            if fam in sb and fam in sd_:
                per_seed_fam[fam]["baseline"].append(sb[fam])
                per_seed_fam[fam]["defended"].append(sd_[fam])
                cells.append(f"{sb[fam]['mean']:6.1f}->{sd_[fam]['mean']:<6.1f}")
            else:
                cells.append(f"{'--':>16}")
        print(f"{seed:>10} " + " ".join(f"{c:>16}" for c in cells))

        for x, y in zip(rb, rd):
            if (
                x["validity_status"] == "valid"
                and x["prediction_drift"] is not None
                and y["prediction_drift"] is not None
            ):
                b_, d_ = abs(x["prediction_drift"]), abs(y["prediction_drift"])
                pooled["baseline"].append(b_)
                pooled["defended"].append(d_)
                paired_diffs.append(b_ - d_)

        results["per_seed"][str(seed)]["families"] = {"baseline": sb, "defended": sd_}
        results["per_seed"][str(seed)]["candidate_sets_identical"] = same

    print(f"\n  matched candidate sets within every seed: {integrity_ok}")

    print("\n--- PER-FAMILY ACROSS SEEDS (mean ± SD of the per-seed statistic) ---")
    print(f"{'family':<15} {'metric':<10} {'baseline':>18} {'defended':>18} {'reduction':>12}")
    print("-" * 78)
    fam_summary = {}
    for fam in FAMILIES:
        if not per_seed_fam[fam]["baseline"]:
            continue
        fam_summary[fam] = {}
        for metric in ("mean", "median", "p95", "max", "frac_above"):
            b = [s[metric] for s in per_seed_fam[fam]["baseline"]]
            d = [s[metric] for s in per_seed_fam[fam]["defended"]]
            red = [x - y for x, y in zip(b, d)]
            fam_summary[fam][metric] = {
                "baseline_mean": float(np.mean(b)), "baseline_sd": float(np.std(b, ddof=1)),
                "defended_mean": float(np.mean(d)), "defended_sd": float(np.std(d, ddof=1)),
                "reduction_mean": float(np.mean(red)), "reduction_sd": float(np.std(red, ddof=1)),
                "reduction_all_seeds_positive": bool(all(r > 0 for r in red)),
            }
            label = "P95" if metric == "p95" else metric
            if metric == "frac_above":
                print(f"{fam:<15} {'>52.02K':<10} "
                      f"{np.mean(b) * 100:>7.1f}% ± {np.std(b, ddof=1) * 100:<8.1f} "
                      f"{np.mean(d) * 100:>7.1f}% ± {np.std(d, ddof=1) * 100:<8.1f} "
                      f"{np.mean(red) * 100:>11.1f}%")
            else:
                print(f"{fam:<15} {label:<10} {msd(b):>18} {msd(d):>18} "
                      f"{np.mean(red):>11.2f}")
        print()
    results["family_summary"] = fam_summary

    # --- Pooled paired ---
    pb, pd_ = np.asarray(pooled["baseline"]), np.asarray(pooled["defended"])
    diffs = np.asarray(paired_diffs)
    print("--- POOLED ACROSS ALL SEEDS (matched candidates) ---")
    print(f"  paired valid candidates    : {diffs.size}")
    print(f"  mean |drift| baseline      : {pb.mean():.2f} K")
    print(f"  mean |drift| defended      : {pd_.mean():.2f} K")
    print(f"  mean reduction             : {diffs.mean():.2f} K")
    print(f"  candidates improved        : {(diffs > 0).mean():.1%}")

    pooled_res = {
        "n_paired": int(diffs.size),
        "baseline_mean": float(pb.mean()), "defended_mean": float(pd_.mean()),
        "mean_reduction": float(diffs.mean()),
        "fraction_improved": float((diffs > 0).mean()),
    }

    # Wilcoxon is appropriate here: paired observations, same candidate scored by
    # both models, no normality assumption. It tests the per-candidate pairing,
    # NOT independence across seeds -- candidates from one seed share a model, so
    # this is reported as a descriptive paired statistic, not evidence about the
    # seed-level population.
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(pb, pd_)
        pooled_res["wilcoxon"] = {
            "statistic": float(stat), "p_value": float(p),
            "caveat": (
                "Paired over candidates, not seeds. Candidates within a seed share "
                "a model, so observations are not fully independent; treat as "
                "descriptive support for the paired difference, not a seed-level claim."
            ),
        }
        print(f"  Wilcoxon signed-rank       : W={stat:.0f}, p={p:.2e}")
        print("    (paired over candidates; NOT a seed-level independence claim)")
    except ImportError:
        pooled_res["wilcoxon"] = "scipy not installed -- test not run"
        print("  Wilcoxon: scipy not installed, test skipped")

    results["pooled"] = pooled_res
    results["integrity"] = {"matched_candidate_sets_all_seeds": integrity_ok}

    out = Path("results/phase2c_audit.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
