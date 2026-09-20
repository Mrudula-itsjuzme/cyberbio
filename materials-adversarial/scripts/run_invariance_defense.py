#!/usr/bin/env python3
"""Preregistered Phase 6 invariance-defense experiment.

Modes deliberately separate bank construction, validation-only selection, and
the one final sealed-test evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import RDLogger
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials_adv.data.scaler import TargetScaler
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.robustness.equivalence import generate_equivalents, is_equivalent
from materials_adv.robustness.repeat_shift import generate_repeat_shift, tokenizer_roundtrip
from materials_adv.training.invariance import (
    anchored_consistency_loss,
    encode_batch,
    objective,
    worst_consistency_loss,
)

OUT = ROOT / "results" / "invariance_defense"
BANKS = OUT / "banks"
MODELS = OUT / "models"
BASELINE = ROOT / "results" / "models" / "transformer_regressor"
SEEDS = (20260815, 20260816, 20260817)
LAMBDA_GRID = (0.01, 0.1, 0.5, 1.0)
MAX_LENGTH = 256
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VAL_CLEAN_GATE = 0.4601476192474365 + 0.02
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_inputs():
    frame = pd.read_csv(ROOT / "data" / "processed" / "processed.csv")
    splits = json.loads((ROOT / "data" / "processed" / "splits.json").read_text())
    vocabulary = json.loads((ROOT / "data" / "processed" / "vocab.json").read_text())
    scaler = TargetScaler.load(BASELINE / "scaler.json")
    return frame, splits, vocabulary, scaler


def build_model(vocabulary, *, positional=True, trainable=True):
    model = TransformerRegressorModel(
        len(vocabulary), 64, 2, 4, 128, 0.1, MAX_LENGTH, "mean",
        use_positional_encoding=positional,
    )
    model.load_state_dict(torch.load(BASELINE / "model.pt", map_location="cpu", weights_only=True))
    if not positional:
        with torch.no_grad():
            model.pos_encoder.weight.zero_()
        model.pos_encoder.weight.requires_grad_(False)
    for parameter in model.parameters():
        if parameter is not model.pos_encoder.weight or positional:
            parameter.requires_grad_(trainable)
    return model


def predict(model, texts, vocabulary, batch_size=64):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids, mask = encode_batch(texts[start:start + batch_size], vocabulary)
            values.extend(model(ids, padding_mask=mask).cpu().numpy())
    return np.asarray(values, dtype=float)


def clean_metrics(model, rows, vocabulary, scaler):
    scaled = predict(model, rows.original_representation.astype(str).tolist(), vocabulary)
    estimates = scaler.inverse_transform(scaled)
    targets = rows.property_value.to_numpy(float)
    residual = estimates - targets
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1 - np.sum(residual ** 2) / np.sum((targets - targets.mean()) ** 2)),
    }, estimates


def valid_broader(source, seed, target=4):
    variants, _ = generate_equivalents(source, seed=seed, target=target, max_attempts=120)
    output = []
    vocabulary = json.loads((ROOT / "data" / "processed" / "vocab.json").read_text())
    for variant in variants:
        normalized = variant.variant_representation.replace("[*]", "__STAR__").replace("*", "[*]").replace("__STAR__", "[*]")
        decision = is_equivalent(source, normalized)
        check = tokenizer_roundtrip(normalized, vocabulary)
        if decision.equivalent and check["unknown_count"] == 0 and check["within_max_length"]:
            output.append((variant.family, normalized, decision.reason))
    return output


def valid_repeat(source, target=4):
    vocabulary = json.loads((ROOT / "data" / "processed" / "vocab.json").read_text())
    output = []
    for variant in generate_repeat_shift(source, cap=target):
        check = tokenizer_roundtrip(variant.variant, vocabulary)
        if variant.equivalence_status == "VERIFIED_EQUIVALENT" and check["unknown_count"] == 0 and check["within_max_length"]:
            output.append((variant.transform_family, variant.variant, variant.equivalence_reason))
    return output


def prepare_banks() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BANKS.mkdir(parents=True, exist_ok=True)
    frame, splits, _, _ = load_inputs()
    test_ids = set(frame.iloc[splits["test"]].polymer_id.astype(str))
    summaries = []
    for split_name, indices in (("train", splits["train"]), ("val", splits["val"])):
        broader_rows, repeat_rows = [], []
        for row_index, row in frame.iloc[indices].iterrows():
            source_id = str(row.polymer_id)
            if source_id in test_ids:
                raise RuntimeError("test source leaked into defense bank")
            source = str(row.original_representation)
            seed = 610000 + int(row_index)
            for number, (family, variant, reason) in enumerate(valid_broader(source, seed)):
                broader_rows.append({
                    "split": split_name, "dataset_row_index": row_index,
                    "source_id": source_id, "family": family, "source": source,
                    "variant": variant, "target": float(row.property_value),
                    "seed": seed, "equivalence_pass": True,
                    "validation_reason": reason, "variant_number": number,
                })
            for number, (family, variant, reason) in enumerate(valid_repeat(source)):
                repeat_rows.append({
                    "split": split_name, "dataset_row_index": row_index,
                    "source_id": source_id, "family": family, "source": source,
                    "variant": variant, "target": float(row.property_value),
                    "seed": seed, "equivalence_pass": True,
                    "validation_reason": reason, "variant_number": number,
                })
        broader = pd.DataFrame(broader_rows)
        repeat = pd.DataFrame(repeat_rows)
        broader.to_csv(BANKS / f"{split_name}_broader.csv", index=False)
        repeat.to_csv(BANKS / f"{split_name}_repeat_shift.csv", index=False)
        summaries.append({"split": split_name, "bank": "broader", "sources": broader.source_id.nunique(), "variants": len(broader)})
        summaries.append({"split": split_name, "bank": "repeat_shift", "sources": repeat.source_id.nunique(), "variants": len(repeat)})
    pd.DataFrame(summaries).to_csv(BANKS / "bank_summary.csv", index=False)
    manifest = {
        "created_before_training": True,
        "split_sha256": sha256(ROOT / "data" / "processed" / "splits.json"),
        "test_cohort_sha256": sha256(ROOT / "results" / "representation_equivalence_sequence" / "cohort_manifest.json"),
        "test_sources_used": 0,
        "max_length": MAX_LENGTH,
        "unknown_allowed": False,
    }
    (BANKS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))


def bank_lookup(bank):
    grouped = defaultdict(list)
    for row in bank.itertuples():
        grouped[str(row.source_id)].append(str(row.variant))
    return grouped


def validation_robustness(model, bank, vocabulary, scaler):
    rows = []
    for source_id, group in bank.groupby("source_id", sort=False):
        texts = [str(group.source.iloc[0]), *group.variant.astype(str).tolist()]
        values = scaler.inverse_transform(predict(model, texts, vocabulary))
        rows.append(float(np.max(np.abs(values[1:] - values[0]))))
    return float(np.mean(rows)) if rows else float("nan")


def train_one(arm, seed, consistency_weight, train_rows, val_rows, train_bank, val_bank, vocabulary, scaler, *, positional=True):
    seed_all(seed)
    model = build_model(vocabulary, positional=positional, trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lookup = bank_lookup(train_bank)
    generator = np.random.default_rng(seed)
    targets_scaled = {
        str(row.polymer_id): float(scaler.transform(float(row.property_value)))
        for row in train_rows.itertuples()
    }
    sources = {str(row.polymer_id): str(row.original_representation) for row in train_rows.itertuples()}
    source_ids = list(sources)
    start_time = time.monotonic()
    steps = source_examples = variant_examples = 0
    candidates = []
    for epoch in range(EPOCHS):
        order = generator.permutation(source_ids)
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            batch_ids = order[start:start + BATCH_SIZE].tolist()
            source_texts = [sources[source_id] for source_id in batch_ids]
            target = torch.tensor([targets_scaled[source_id] for source_id in batch_ids], dtype=torch.float32)
            source_tensor, source_mask = encode_batch(source_texts, vocabulary)
            source_prediction = model(source_tensor, padding_mask=source_mask)
            property_loss = F.mse_loss(source_prediction, target)
            count = 4 if arm == "broader_worst" else 1
            variant_texts = []
            for offset, source_id in enumerate(batch_ids):
                available = lookup.get(source_id) or [sources[source_id]]
                chosen = [available[(epoch + offset + j) % len(available)] for j in range(count)]
                variant_texts.extend(chosen)
            variant_tensor, variant_mask = encode_batch(variant_texts, vocabulary)
            variant_prediction = model(variant_tensor, padding_mask=variant_mask).reshape(len(batch_ids), count)
            if arm == "ordinary_augmentation":
                expanded_target = target.unsqueeze(1).expand_as(variant_prediction)
                loss = 0.5 * (property_loss + F.mse_loss(variant_prediction, expanded_target))
            else:
                consistency = worst_consistency_loss(source_prediction, variant_prediction) if arm == "broader_worst" else anchored_consistency_loss(source_prediction, variant_prediction)
                loss = objective(property_loss, consistency, consistency_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            source_examples += len(batch_ids)
            variant_examples += len(variant_texts)
        clean, _ = clean_metrics(model, val_rows, vocabulary, scaler)
        robust = validation_robustness(model, val_bank, vocabulary, scaler)
        candidates.append({
            "epoch": epoch + 1, "val_mae": clean["mae"],
            "val_robustness": robust,
            "state": copy.deepcopy(model.state_dict()),
        })
    eligible = [item for item in candidates if item["val_mae"] <= VAL_CLEAN_GATE]
    pool = eligible or candidates
    best = sorted(pool, key=lambda item: (item["val_robustness"] if eligible else item["val_mae"], item["val_mae"]))[0]
    model.load_state_dict(best["state"])
    accounting = {
        "optimizer_steps": steps, "source_examples": source_examples,
        "variant_examples": variant_examples, "epochs": EPOCHS,
        "wall_seconds": time.monotonic() - start_time,
        "selected_epoch": best["epoch"], "val_mae": best["val_mae"],
        "val_robustness": best["val_robustness"],
        "clean_gate_pass": best["val_mae"] <= VAL_CLEAN_GATE,
    }
    return model, accounting


def selection() -> None:
    if not (BANKS / "manifest.json").exists():
        raise RuntimeError("prepare banks before lambda selection")
    frame, splits, vocabulary, scaler = load_inputs()
    train_rows, val_rows = frame.iloc[splits["train"]], frame.iloc[splits["val"]]
    arms = {
        "repeat_shift_average": "repeat_shift",
        "broader_average": "broader",
        "broader_worst": "broader",
    }
    records, selected = [], {}
    MODELS.mkdir(parents=True, exist_ok=True)
    for arm, bank_name in arms.items():
        train_bank = pd.read_csv(BANKS / f"train_{bank_name}.csv")
        val_bank = pd.read_csv(BANKS / f"val_{bank_name}.csv")
        arm_runs = []
        for weight in LAMBDA_GRID:
            print(f"selection {arm} lambda={weight}", flush=True)
            model, account = train_one(arm, SEEDS[0], weight, train_rows, val_rows, train_bank, val_bank, vocabulary, scaler)
            record = {"arm": arm, "seed": SEEDS[0], "lambda": weight, **account}
            records.append(record)
            arm_runs.append((record, copy.deepcopy(model.state_dict())))
        eligible = [item for item in arm_runs if item[0]["clean_gate_pass"]]
        pool = eligible or arm_runs
        best_record, best_state = sorted(
            pool,
            key=lambda item: (
                item[0]["val_robustness"] if eligible else item[0]["val_mae"],
                item[0]["val_mae"], item[0]["lambda"],
            ),
        )[0]
        selected[arm] = best_record["lambda"]
        path = MODELS / arm / str(SEEDS[0])
        path.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, path / "model.pt")
        (path / "training.json").write_text(json.dumps(best_record, indent=2) + "\n")
    pd.DataFrame(records).to_csv(OUT / "lambda_selection.csv", index=False)
    manifest = {
        "selected_lambdas": selected,
        "selection_split": "validation",
        "selection_seed": SEEDS[0],
        "test_accessed": False,
        "clean_gate_eV": VAL_CLEAN_GATE,
    }
    (OUT / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def evaluate_threat(model, variants, vocabulary, scaler, threat):
    rows = []
    variant_col = "variant"
    source_col = "source"
    for source_id, group in variants.groupby("source_id", sort=False):
        texts = [str(group[source_col].iloc[0]), *group[variant_col].astype(str).tolist()]
        values = scaler.inverse_transform(predict(model, texts, vocabulary))
        for row, value in zip(group.itertuples(), values[1:]):
            rows.append({
                "source_id": source_id, "variant_id": row.variant_id,
                "family": row.family, "threat": threat,
                "prediction_source": values[0], "prediction_variant": value,
                "absolute_drift": abs(value - values[0]),
                "position_displacement": getattr(row, "position_displacement", np.nan),
                "token_count_delta": getattr(row, "token_count_delta", np.nan),
            })
    detail = pd.DataFrame(rows)
    source = detail.groupby("source_id").absolute_drift.agg(["max", "mean", "median"]).reset_index()
    source = source.rename(columns={"max": "worst_case", "mean": "mean_drift", "median": "median_drift"})
    return detail, source


def load_test_banks():
    broad = pd.read_csv(ROOT / "results" / "representation_equivalence_validation" / "token_pair_audit.csv")
    broad = broad.rename(columns={"source_representation": "source", "strict_variant": "variant"})
    broad["token_count_delta"] = broad.strict_variant_tokens - broad.source_tokens
    broad = broad[["source_id", "variant_id", "family", "source", "variant", "position_displacement", "token_count_delta"]]
    repeat = pd.read_csv(ROOT / "results" / "repeat_shift_invariance" / "variants.csv")
    repeat = repeat.rename(columns={"source_representation": "source", "variant_representation": "variant"})
    repeat["token_count_delta"] = repeat.token_count_variant - repeat.token_count_source
    repeat = repeat[["source_id", "variant_id", "family", "source", "variant", "position_displacement", "token_count_delta"]]
    return repeat, broad


def bootstrap_paired(baseline, defended, seed, repetitions=5000):
    merged = baseline[["source_id", "worst_case"]].merge(
        defended[["source_id", "worst_case"]], on="source_id", suffixes=("_baseline", "_defense")
    )
    change = merged.worst_case_baseline - merged.worst_case_defense
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    values = change.to_numpy()
    for index in range(repetitions):
        means[index] = rng.choice(values, len(values), replace=True).mean()
    statistic, p_value = wilcoxon(values, alternative="greater")
    return {
        "n_sources": len(values), "mean_reduction": float(values.mean()),
        "median_reduction": float(np.median(values)),
        "bootstrap_ci_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci_high": float(np.quantile(means, 0.975)),
        "wilcoxon_statistic": float(statistic), "wilcoxon_p": float(p_value),
        "standardized_mean_change": float(values.mean() / values.std(ddof=1)),
    }


def final_evaluation() -> None:
    selection_path = OUT / "selection_manifest.json"
    if not selection_path.exists():
        raise RuntimeError("validation selection must be frozen before test evaluation")
    selected = json.loads(selection_path.read_text())["selected_lambdas"]
    frame, splits, vocabulary, scaler = load_inputs()
    train_rows, val_rows, test_rows = frame.iloc[splits["train"]], frame.iloc[splits["val"]], frame.iloc[splits["test"]]
    run_records = []
    arms = {
        "ordinary_augmentation": ("broader", 0.0),
        "repeat_shift_average": ("repeat_shift", selected["repeat_shift_average"]),
        "broader_average": ("broader", selected["broader_average"]),
        "broader_worst": ("broader", selected["broader_worst"]),
    }
    for arm, (bank_name, weight) in arms.items():
        train_bank = pd.read_csv(BANKS / f"train_{bank_name}.csv")
        val_bank = pd.read_csv(BANKS / f"val_{bank_name}.csv")
        for seed in SEEDS:
            path = MODELS / arm / str(seed)
            if (path / "model.pt").exists():
                continue
            print(f"final training {arm} seed={seed} lambda={weight}", flush=True)
            model, account = train_one(arm, seed, weight, train_rows, val_rows, train_bank, val_bank, vocabulary, scaler)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path / "model.pt")
            record = {"arm": arm, "seed": seed, "lambda": weight, "positional": True, **account}
            (path / "training.json").write_text(json.dumps(record, indent=2) + "\n")
    # Held-out branch ordering: train only on start-point and randomized order.
    subset_train = pd.read_csv(BANKS / "train_broader.csv")
    subset_val = pd.read_csv(BANKS / "val_broader.csv")
    allowed = ["start_point_shift", "randomized_atom_ordering"]
    subset_train = subset_train[subset_train.family.isin(allowed)]
    subset_val_train = subset_val[subset_val.family.isin(allowed)]
    unseen_path = MODELS / "unseen_family" / str(SEEDS[0])
    if not (unseen_path / "model.pt").exists():
        model, account = train_one("broader_average", SEEDS[0], selected["broader_average"], train_rows, val_rows, subset_train, subset_val_train, vocabulary, scaler)
        unseen_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), unseen_path / "model.pt")
        (unseen_path / "training.json").write_text(json.dumps(account, indent=2) + "\n")
    # Mechanism ablation: same broader consistency, no positional embeddings.
    no_pos_path = MODELS / "broader_average_no_position" / str(SEEDS[0])
    if not (no_pos_path / "model.pt").exists():
        model, account = train_one("broader_average", SEEDS[0], selected["broader_average"], train_rows, val_rows, pd.read_csv(BANKS / "train_broader.csv"), pd.read_csv(BANKS / "val_broader.csv"), vocabulary, scaler, positional=False)
        no_pos_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), no_pos_path / "model.pt")
        (no_pos_path / "training.json").write_text(json.dumps(account, indent=2) + "\n")

    repeat_bank, broad_bank = load_test_banks()
    evaluation_models = {"baseline": build_model(vocabulary, trainable=False)}
    for arm in arms:
        for seed in SEEDS:
            model = build_model(vocabulary)
            model.load_state_dict(torch.load(MODELS / arm / str(seed) / "model.pt", map_location="cpu", weights_only=True))
            evaluation_models[f"{arm}_seed{seed}"] = model
    for name, path, positional in (
        ("unseen_family_seed20260815", unseen_path / "model.pt", True),
        ("broader_average_no_position_seed20260815", no_pos_path / "model.pt", False),
    ):
        model = build_model(vocabulary, positional=positional)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        evaluation_models[name] = model

    clean_records, repeat_records, broad_records, family_records = [], [], [], []
    source_cache, detail_cache = {}, {}
    for model_name, model in evaluation_models.items():
        metrics, _ = clean_metrics(model, test_rows, vocabulary, scaler)
        clean_records.append({"model": model_name, **metrics})
        for threat, bank in (("repeat_shift", repeat_bank), ("broader", broad_bank)):
            detail, source = evaluate_threat(model, bank, vocabulary, scaler, threat)
            source_cache[(model_name, threat)] = source
            detail_cache[(model_name, threat)] = detail
            summary = {
                "model": model_name, "n_sources": source.source_id.nunique(),
                "n_variants": len(detail), "mean_source_worst": source.worst_case.mean(),
                "median_source_worst": source.worst_case.median(),
                "P90_source_worst": source.worst_case.quantile(0.9),
                "P95_source_worst": source.worst_case.quantile(0.95),
                "max_source_worst": source.worst_case.max(),
            }
            (repeat_records if threat == "repeat_shift" else broad_records).append(summary)
            for family, group in detail.groupby("family"):
                worst = group.groupby("source_id").absolute_drift.max()
                family_records.append({
                    "model": model_name, "threat": threat, "family": family,
                    "n_sources": group.source_id.nunique(), "n_variants": len(group),
                    "mean_source_worst": worst.mean(), "median_source_worst": worst.median(),
                    "P95_variant_drift": group.absolute_drift.quantile(0.95),
                    "max_variant_drift": group.absolute_drift.max(),
                })

    clean = pd.DataFrame(clean_records)
    repeat_summary = pd.DataFrame(repeat_records)
    broad_summary = pd.DataFrame(broad_records)
    families = pd.DataFrame(family_records)
    clean.to_csv(OUT / "clean_metrics.csv", index=False)
    repeat_summary.to_csv(OUT / "repeat_shift_robustness.csv", index=False)
    broad_summary.to_csv(OUT / "broader_equivalence_robustness.csv", index=False)
    families.to_csv(OUT / "family_robustness.csv", index=False)

    primary_rows, stats_rows, tradeoff_rows = [], [], []
    baseline_clean = clean[clean.model == "baseline"].iloc[0]
    baseline_broad = broad_summary[broad_summary.model == "baseline"].iloc[0]
    for arm in arms:
        names = [f"{arm}_seed{seed}" for seed in SEEDS]
        arm_clean = clean[clean.model.isin(names)]
        arm_repeat = repeat_summary[repeat_summary.model.isin(names)]
        arm_broad = broad_summary[broad_summary.model.isin(names)]
        primary_rows.append({
            "model": arm, "seeds": 3,
            "clean_mae_mean": arm_clean.mae.mean(), "clean_mae_std": arm_clean.mae.std(ddof=1),
            "repeat_worst_mean": arm_repeat.mean_source_worst.mean(),
            "repeat_worst_seed_std": arm_repeat.mean_source_worst.std(ddof=1),
            "broader_worst_mean": arm_broad.mean_source_worst.mean(),
            "broader_worst_seed_std": arm_broad.mean_source_worst.std(ddof=1),
        })
        tradeoff_rows.append({
            "model": arm, "delta_clean_mae": arm_clean.mae.mean() - baseline_clean.mae,
            "delta_broader_worst": arm_broad.mean_source_worst.mean() - baseline_broad.mean_source_worst,
            "broader_worst_reduction": baseline_broad.mean_source_worst - arm_broad.mean_source_worst.mean(),
        })
        for seed in SEEDS:
            name = f"{arm}_seed{seed}"
            stats_rows.append({"model": arm, "seed": seed, "threat": "broader", **bootstrap_paired(source_cache[("baseline", "broader")], source_cache[(name, "broader")], seed)})
            stats_rows.append({"model": arm, "seed": seed, "threat": "repeat_shift", **bootstrap_paired(source_cache[("baseline", "repeat_shift")], source_cache[(name, "repeat_shift")], seed)})
    pd.DataFrame(primary_rows).to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(tradeoff_rows).to_csv(OUT / "clean_robust_tradeoff.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(OUT / "statistical_results.csv", index=False)

    branch = families[(families.threat == "broader") & (families.family == "branch_order_permutation")]
    branch.to_csv(OUT / "unseen_family_generalization.csv", index=False)

    shortcut_rows = []
    for model_name in evaluation_models:
        for threat in ("repeat_shift", "broader"):
            detail = detail_cache[(model_name, threat)]
            for feature in ("position_displacement", "token_count_delta"):
                valid = detail[[feature, "absolute_drift"]].dropna()
                rho, p_value = spearmanr(valid[feature], valid.absolute_drift) if len(valid) > 2 else (np.nan, np.nan)
                shortcut_rows.append({"model": model_name, "threat": threat, "feature": feature, "spearman_rho": rho, "p_value": p_value, "n_variants": len(valid)})
    pd.DataFrame(shortcut_rows).to_csv(OUT / "shortcut_analysis.csv", index=False)

    old = pd.read_csv(ROOT / "results" / "representation_equivalence_sequence" / "predictions.csv")
    strict = pd.read_csv(ROOT / "results" / "representation_equivalence_validation" / "token_pair_audit.csv")
    tokenizer = pd.DataFrame([{
        "intervention": "historical_unknown_star_to_padding",
        "mean_drift": old.absolute_drift.mean(), "median_drift": old.absolute_drift.median(),
        "P95_drift": old.absolute_drift.quantile(0.95), "max_drift": old.absolute_drift.max(),
    }, {
        "intervention": "normalized_attachment_token_strict",
        "mean_drift": strict.absolute_drift.mean(), "median_drift": strict.absolute_drift.median(),
        "P95_drift": strict.absolute_drift.quantile(0.95), "max_drift": strict.absolute_drift.max(),
    }])
    tokenizer.to_csv(OUT / "tokenizer_fix_comparison.csv", index=False)

    # Real-data figures.
    plot_models = ["baseline", *[f"broader_average_seed{seed}" for seed in SEEDS]]
    plot = broad_summary[broad_summary.model.isin(plot_models)]
    plot.plot.bar(x="model", y="mean_source_worst", legend=False); plt.ylabel("mean source worst drift (eV)"); plt.tight_layout(); plt.savefig(OUT / "baseline_vs_defense_worst_case.png", dpi=160); plt.close()
    trade = pd.DataFrame(tradeoff_rows); plt.scatter(trade.delta_clean_mae, trade.broader_worst_reduction); [plt.annotate(row.model, (row.delta_clean_mae, row.broader_worst_reduction)) for row in trade.itertuples()]; plt.xlabel("clean MAE change (eV)"); plt.ylabel("broader worst-drift reduction (eV)"); plt.tight_layout(); plt.savefig(OUT / "clean_vs_robust_tradeoff.png", dpi=160); plt.close()
    for threat, table, filename in (("repeat_shift", repeat_summary, "repeat_shift_defense.png"), ("broader", broad_summary, "broader_equivalence_defense.png")):
        table[table.model.isin(plot_models)].plot.bar(x="model", y="mean_source_worst", legend=False); plt.ylabel(f"{threat} mean source worst (eV)"); plt.tight_layout(); plt.savefig(OUT / filename, dpi=160); plt.close()
    families[families.model.isin(plot_models)].pivot(index="family", columns="model", values="mean_source_worst").plot.bar(); plt.ylabel("mean source worst (eV)"); plt.tight_layout(); plt.savefig(OUT / "family_robustness.png", dpi=160); plt.close()
    before = source_cache[("baseline", "broader")].sort_values("source_id"); after = source_cache[(f"broader_average_seed{SEEDS[0]}", "broader")].sort_values("source_id"); plt.scatter(before.worst_case, after.worst_case, s=12); lim=max(before.worst_case.max(),after.worst_case.max()); plt.plot([0,lim],[0,lim],'k--'); plt.xlabel("baseline"); plt.ylabel("defense"); plt.tight_layout(); plt.savefig(OUT / "source_level_before_after.png", dpi=160); plt.close()
    plt.hist(before.worst_case, alpha=.5, label="baseline"); plt.hist(after.worst_case, alpha=.5, label="defense"); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "worst_case_distribution_before_after.png", dpi=160); plt.close()
    branch.pivot(index="family", columns="model", values="mean_source_worst").plot.bar(); plt.tight_layout(); plt.savefig(OUT / "unseen_family_generalization.png", dpi=160); plt.close()
    shortcuts = pd.DataFrame(shortcut_rows); shortcuts[(shortcuts.feature=="position_displacement") & shortcuts.model.isin(plot_models)].plot.bar(x="model",y="spearman_rho",legend=False); plt.tight_layout(); plt.savefig(OUT / "positional_sensitivity_before_after.png",dpi=160); plt.close()
    tokenizer.plot.bar(x="intervention",y="mean_drift",legend=False); plt.tight_layout(); plt.savefig(OUT / "tokenizer_fix_vs_invariance_training.png",dpi=160); plt.close()

    accounting = []
    for path in MODELS.glob("*/*/training.json"):
        record = json.loads(path.read_text())
        accounting.append({"model": path.parents[1].name, "seed": path.parent.name, **record})
    pd.DataFrame(accounting).to_csv(OUT / "training_accounting.csv", index=False)
    final_manifest = {"test_evaluated_once": True, "selection_manifest_sha256": sha256(selection_path), "models": len(evaluation_models)}
    (OUT / "final_evaluation_manifest.json").write_text(json.dumps(final_manifest, indent=2) + "\n")
    print(pd.DataFrame(primary_rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "select", "final"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_banks()
    elif args.mode == "select":
        selection()
    else:
        final_evaluation()


if __name__ == "__main__":
    main()
