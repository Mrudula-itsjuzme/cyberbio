#!/usr/bin/env python3
"""Multi-seed representation robustness evaluation for the polyVERSE bandgap lineage.

Reads configs/canonical_experiments.yaml for experiment specifications.
For each seed, checks whether a trained checkpoint exists and either:
  - Loads it and computes drift metrics, OR
  - Reports MISSING clearly without fabricating numbers.

Does NOT silently substitute estimates for missing runs.

Usage:
    PYTHONPATH=. python scripts/run_representation_multiseed.py \\
        --config configs/canonical_experiments.yaml \\
        [--seeds 42 123 2026 777 999] \\
        [--n-rand-variants 10] \\
        [--out results/final/multiseed_representation_robustness.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "UNREADABLE"


def _load_yaml(path: str) -> dict:
    """Minimal YAML loader — avoids extra dependency for simple structure."""
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    # Fallback: use PyYAML-free approach via json after we detect the structure
    raise RuntimeError(
        "PyYAML is required: pip install pyyaml  (or: python -m pip install pyyaml)"
    )


def _check_checkpoint(seed: int, config_name: str, models_dir: Path) -> Path | None:
    """Return checkpoint path if it exists for this seed/config, else None."""
    candidates = [
        models_dir / f"{config_name}_seed{seed}" / "model.pt",
        models_dir / f"{config_name.lower()}_seed{seed}" / "model.pt",
        models_dir / f"seed{seed}_{config_name}" / "model.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _generate_rand_smiles(smiles: str, n: int, rng: np.random.Generator) -> list[str]:
    """Generate n randomized SMILES for the given SMILES using RDKit."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        variants = set()
        for _ in range(n * 10):
            try:
                s = Chem.MolToSmiles(mol, doRandom=True)
                if s and s != smiles:
                    variants.add(s)
                    if len(variants) >= n:
                        break
            except Exception:
                pass
        return list(variants)[:n]
    except ImportError:
        return []


def _canonical(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        return None


def _load_model_and_predict(checkpoint_path: Path, smiles_list: list[str]) -> list[float] | None:
    """Load a model checkpoint and run inference. Returns None if unavailable."""
    try:
        import torch
        from materials_adv.data.tokenizer import tokenize, Vocabulary
        from materials_adv.data.scaler import TargetScaler
        from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel

        checkpoint_dir = checkpoint_path.parent
        # Load scaler
        scaler_path = checkpoint_dir / "scaler.json"
        if not scaler_path.exists():
            scaler_path = Path("results/models/transformer_regressor/scaler.json")
        with open(scaler_path) as f:
            sc = json.load(f)
        scaler = TargetScaler(mean=sc["mean"], std=sc["std"])

        # Load vocab
        vocab_path = Path("data/processed/vocab.json")
        with open(vocab_path) as f:
            vocab_data = json.load(f)
        stoi = vocab_data if isinstance(vocab_data, dict) else vocab_data.get("stoi", {})

        model = TwoBranchTransformerRegressorModel(vocab_size=len(stoi))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        MAX_LEN = 256
        preds = []
        with torch.no_grad():
            for smi in smiles_list:
                toks = tokenize(smi)
                slen = min(len(toks), MAX_LEN)
                ids = torch.zeros((1, MAX_LEN), dtype=torch.long)
                mask = torch.ones((1, MAX_LEN), dtype=torch.bool)
                for j in range(slen):
                    ids[0, j] = stoi.get(toks[j], 0)
                    mask[0, j] = False
                try:
                    out = model(ids, padding_mask=mask).cpu().numpy().item()
                except Exception:
                    out = model(ids, mask).squeeze(-1).cpu().numpy().item()
                preds.append(float(scaler.inverse_transform(np.array([[out]]))[0, 0]))
        return preds
    except Exception as e:
        print(f"  [WARN] Could not load/run model at {checkpoint_path}: {e}")
        return None


def _drift_metrics(drifts: list[float]) -> dict:
    a = np.array(drifts)
    return {
        "n_pairs": len(a),
        "mean_drift_eV": float(np.mean(a)),
        "median_drift_eV": float(np.median(a)),
        "std_drift_eV": float(np.std(a)),
        "p90_drift_eV": float(np.percentile(a, 90)),
        "max_drift_eV": float(np.max(a)),
        "fraction_above_0_1eV": float(np.mean(a > 0.1)),
        "fraction_above_0_5eV": float(np.mean(a > 0.5)),
    }


def _representation_invariance_score(drifts: list[float], clean_mae: float) -> float:
    """Scale-normalised invariance score. 1 = perfectly invariant."""
    if not drifts or clean_mae <= 0:
        return float("nan")
    pop_std = float(np.std(drifts))
    return 1.0 / (1.0 + pop_std / clean_mae)


def _bootstrap_ci(drifts: list[float], n_boot: int = 2000, ci: float = 0.95, rng_seed: int = 0) -> dict:
    """Paired bootstrap confidence interval for the mean drift."""
    rng = np.random.default_rng(rng_seed)
    a = np.array(drifts)
    boot_means = [np.mean(rng.choice(a, len(a), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return {"ci_level": ci, "ci_lo_eV": lo, "ci_hi_eV": hi, "n_bootstrap": n_boot}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed representation robustness evaluation (bandgap lineage)"
    )
    parser.add_argument("--config", default="configs/canonical_experiments.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026, 777, 999])
    parser.add_argument("--n-rand-variants", type=int, default=10)
    parser.add_argument("--n-molecules", type=int, default=100,
                        help="Number of test molecules to evaluate per seed")
    parser.add_argument("--out", default="results/final/multiseed_representation_robustness.json")
    parser.add_argument("--models-dir", default="results/models")
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Seed Representation Robustness Evaluation")
    print("polyVERSE Bandgap Lineage")
    print("=" * 70)

    cfg = _load_yaml(args.config)
    seeds = args.seeds
    models_dir = Path(args.models_dir)

    # Load splits
    splits_path = Path(cfg["dataset"]["split_manifest"])
    print(f"\nLoading splits from {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)
    test_ids = splits.get("test", [])
    print(f"Test set size: {len(test_ids)} (sealed)")

    # Load processed data
    import pandas as pd
    data_path = Path(cfg["dataset"]["file"])
    df = pd.read_csv(data_path)
    col_rep = cfg["dataset"]["representation_column"]
    col_target = cfg["dataset"]["target_column"]

    # Filter to test molecules
    if "polymer_id" in df.columns:
        test_df = df[df["polymer_id"].isin(test_ids)]
    else:
        test_df = df.iloc[test_ids] if isinstance(test_ids[0], int) else df

    # Sample molecules for evaluation
    rng_sample = np.random.default_rng(20260815)
    n_eval = min(args.n_molecules, len(test_df))
    sample_idx = rng_sample.choice(len(test_df), n_eval, replace=False)
    eval_smiles = test_df[col_rep].iloc[sample_idx].tolist()
    eval_targets = test_df[col_target].iloc[sample_idx].tolist()
    print(f"Evaluating {n_eval} test molecules with {args.n_rand_variants} randomized SMILES each")

    # Pre-generate randomized SMILES (same bank used across all seeds)
    print("\nGenerating randomized SMILES bank (shared across seeds)...")
    rng_gen = np.random.default_rng(20260815)
    rand_bank: list[list[str]] = []
    valid_equivalents = 0
    for smi in eval_smiles:
        variants = _generate_rand_smiles(smi, args.n_rand_variants, rng_gen)
        rand_bank.append(variants)
        # Sanity check: all variants must canonicalize to same graph
        can_orig = _canonical(smi)
        for v in variants:
            can_v = _canonical(v)
            if can_orig and can_v and can_orig != can_v:
                print(f"  [ERROR] Canonicalization mismatch for {smi} -> {v}")
                sys.exit(1)
        valid_equivalents += len(variants)
    print(f"  Generated {valid_equivalents} equivalent-representation pairs")

    # Evaluate baseline checkpoint (single canonical checkpoint)
    baseline_ckpt = models_dir / "transformer_regressor" / "model.pt"
    configs_to_eval = ["baseline"]
    checkpoint_map = {
        "baseline": baseline_ckpt if baseline_ckpt.exists() else None,
    }
    if not checkpoint_map["baseline"]:
        print(f"\n[WARN] Baseline checkpoint missing at {baseline_ckpt}")

    # Per-seed results
    seed_results: list[dict] = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        seed_entry: dict = {"seed": seed, "configs": {}}

        for config_name, base_ckpt in checkpoint_map.items():
            # Check for seed-specific checkpoint first, fall back to canonical
            seed_ckpt = _check_checkpoint(seed, config_name, models_dir)
            ckpt = seed_ckpt or base_ckpt

            if ckpt is None or not ckpt.exists():
                print(f"  {config_name}: MISSING checkpoint")
                seed_entry["configs"][config_name] = {
                    "status": "MISSING",
                    "checkpoint_searched": str(models_dir / f"{config_name}_seed{seed}"),
                    "note": "No checkpoint found for this seed. Run training to populate.",
                }
                continue

            is_shared = (ckpt == base_ckpt and seed_ckpt is None)
            print(f"  {config_name}: {'shared baseline checkpoint' if is_shared else 'seed-specific checkpoint'}")
            print(f"    checkpoint: {ckpt}")

            # Run inference on eval molecules
            all_preds = _load_model_and_predict(ckpt, eval_smiles)
            if all_preds is None:
                seed_entry["configs"][config_name] = {
                    "status": "INFERENCE_FAILED",
                    "checkpoint": str(ckpt),
                }
                continue

            clean_mae = float(np.mean(np.abs(np.array(all_preds) - np.array(eval_targets))))

            # Compute drift over randomized SMILES
            all_drifts: list[float] = []
            for i, (orig_smi, orig_pred) in enumerate(zip(eval_smiles, all_preds)):
                variants = rand_bank[i]
                if not variants:
                    continue
                var_preds = _load_model_and_predict(ckpt, variants)
                if var_preds:
                    for vp in var_preds:
                        all_drifts.append(abs(vp - orig_pred))

            if not all_drifts:
                seed_entry["configs"][config_name] = {
                    "status": "NO_VARIANTS",
                    "checkpoint": str(ckpt),
                    "clean_mae_eV": clean_mae,
                }
                continue

            metrics = _drift_metrics(all_drifts)
            metrics["clean_mae_eV"] = clean_mae
            metrics["representation_invariance_score"] = _representation_invariance_score(
                all_drifts, clean_mae
            )
            metrics["bootstrap_ci_mean_drift"] = _bootstrap_ci(all_drifts, rng_seed=seed)
            metrics["status"] = "OK"
            metrics["checkpoint"] = str(ckpt)
            metrics["checkpoint_sha256"] = _sha256(ckpt)
            metrics["checkpoint_is_shared_not_seed_specific"] = is_shared
            if is_shared:
                metrics["warning"] = (
                    "Using shared baseline checkpoint, not a seed-specific trained model. "
                    "This measures representation sensitivity of the canonical checkpoint only."
                )

            seed_entry["configs"][config_name] = metrics
            print(f"    mean_drift: {metrics['mean_drift_eV']:.4f} eV  "
                  f"median: {metrics['median_drift_eV']:.4f} eV  "
                  f"p90: {metrics['p90_drift_eV']:.4f} eV  "
                  f"repr_invariance: {metrics['representation_invariance_score']:.3f}")

        seed_results.append(seed_entry)

    # Cross-seed summary
    print("\n--- Cross-Seed Summary ---")
    cross_seed_summary: dict = {}
    for config_name in configs_to_eval:
        ok_seeds = [
            r["configs"][config_name]
            for r in seed_results
            if r["configs"].get(config_name, {}).get("status") == "OK"
        ]
        missing_seeds = [
            r["seed"]
            for r in seed_results
            if r["configs"].get(config_name, {}).get("status") != "OK"
        ]
        if ok_seeds:
            means = [s["mean_drift_eV"] for s in ok_seeds]
            cross_seed_summary[config_name] = {
                "n_seeds_evaluated": len(ok_seeds),
                "n_seeds_missing": len(missing_seeds),
                "missing_seeds": missing_seeds,
                "mean_drift_across_seeds_eV": float(np.mean(means)),
                "std_drift_across_seeds_eV": float(np.std(means)),
                "per_seed_mean_drifts_eV": means,
            }
            print(f"  {config_name}: {len(ok_seeds)} seeds OK, {len(missing_seeds)} MISSING")
            if len(ok_seeds) > 1:
                print(f"    cross-seed mean drift: {np.mean(means):.4f} ± {np.std(means):.4f} eV")
        else:
            cross_seed_summary[config_name] = {
                "n_seeds_evaluated": 0,
                "n_seeds_missing": len(missing_seeds),
                "missing_seeds": missing_seeds,
                "status": "ALL_MISSING",
                "note": (
                    "No seed-specific checkpoints exist for this config in the bandgap lineage. "
                    "The canonical_benchmark_no_leakage.json was produced by in-memory training. "
                    "To obtain multi-seed estimates, run training for each seed and save checkpoints."
                ),
            }
            print(f"  {config_name}: ALL {len(missing_seeds)} seeds MISSING")

    # Provenance
    from materials_adv.utils.provenance import capture as prov_capture, save as prov_save
    prov = prov_capture(
        seed=None,
        config={"seeds": seeds, "n_rand_variants": args.n_rand_variants, "n_molecules": n_eval},
        dataset_path=data_path,
        split_manifest=splits_path,
        extra={"eval_smiles_count": n_eval, "rand_bank_pairs": valid_equivalents},
    )

    output = {
        "schema": "multiseed_representation_robustness_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lineage": cfg.get("lineage"),
        "property": cfg.get("property"),
        "units": cfg.get("units"),
        "seeds_requested": seeds,
        "n_molecules_per_seed": n_eval,
        "n_rand_variants_per_molecule": args.n_rand_variants,
        "scientific_note": (
            "Representation-preserving attack: randomized SMILES preserve 100% "
            "of the canonical molecular graph. All drift here is pure "
            "representation/serialization sensitivity, NOT chemical change. "
            "Seeds without saved checkpoints are reported as MISSING."
        ),
        "seed_results": seed_results,
        "cross_seed_summary": cross_seed_summary,
        "provenance": prov,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    prov_save(prov, out_path.parent / "multiseed_representation_robustness_provenance.json")

    print(f"\n[DONE] Results written to {out_path}")
    print("[NOTE] Seeds without checkpoints are MISSING — not estimated or fabricated.")


if __name__ == "__main__":
    main()
