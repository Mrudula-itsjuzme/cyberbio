#!/usr/bin/env python3
"""Audit verified equivalent PSMILES on the frozen canonical sequence checkpoint."""

from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.robustness.equivalence import generate_equivalents, is_equivalent

SEED = 20260920
OUT = ROOT / "results/representation_equivalence_sequence"
MODEL_DIR = ROOT / "results/models/transformer_regressor"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_model():
    vocab = json.loads((ROOT / "data/processed/vocab.json").read_text())
    model = TransformerRegressorModel(len(vocab), 64, 2, 4, 128, 0.1, 256, "mean")
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    predictor = TransformerRegressor(model, vocab, TargetScaler.load(MODEL_DIR / "scaler.json"), "eV")
    return model, predictor, vocab


def reproduce_history(predictor, frame: pd.DataFrame, splits: dict) -> dict:
    historical_path = ROOT / "results/representation_attribution/baseline_predictions.csv"
    historical = pd.read_csv(historical_path)
    current_rows = frame.iloc[splits["val"]]
    if historical.original_representation.tolist() != current_rows.original_representation.tolist():
        raise RuntimeError("CHECKPOINT_REPRODUCTION_FAILURE: historical rows do not match frozen validation split")
    current = predictor.predict(current_rows.original_representation.tolist())
    stored = historical.clean_transformer.to_numpy(float)
    delta = np.abs(current - stored)
    tolerance = 1e-6
    if not np.allclose(current, stored, rtol=0.0, atol=tolerance):
        raise RuntimeError(f"CHECKPOINT_REPRODUCTION_FAILURE: max prediction difference {delta.max()}")
    return {"status": "PASS", "comparison": str(historical_path.relative_to(ROOT)),
            "n": len(current), "absolute_tolerance": tolerance,
            "max_absolute_difference": float(delta.max())}


def bootstrap_rho(x, y, seed=SEED, n=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed); values = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r): values.append(r)
    return rho, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def savefig(name):
    plt.tight_layout(); plt.savefig(OUT / name, dpi=160); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--sources", type=int, default=100)
    parser.add_argument("--variants", type=int, default=30); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data/processed/processed.csv")
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    model, predictor, vocab = load_frozen_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    reproduction = reproduce_history(predictor, frame, splits)
    model_manifest = {
        "model_type": "TransformerRegressorModel", "architecture": "two-layer Transformer encoder",
        "property": "polyVERSE polymer Bandgap", "units": "eV",
        "checkpoint": str((MODEL_DIR / 'model.pt').relative_to(ROOT)),
        "checkpoint_sha256": sha(MODEL_DIR / "model.pt"),
        "scaler": str((MODEL_DIR / 'scaler.json').relative_to(ROOT)),
        "scaler_sha256": sha(MODEL_DIR / "scaler.json"),
        "tokenizer_source": "src/materials_adv/data/tokenizer.py:tokenize",
        "vocabulary_path": "data/processed/vocab.json", "vocabulary_sha256": sha(ROOT / "data/processed/vocab.json"),
        "vocabulary_size": len(vocab), "max_sequence_length": 256,
        "padding": "dynamic per inference batch; right padding with id 0 and boolean padding mask",
        "truncation": "none; sequences longer than 256 raise ValueError",
        "unknown_token_behavior": "tokens absent from frozen vocabulary map to id 0 (same index as padding in historical wrapper)",
        "special_tokens": "none inserted during canonical inference; embedding index 0 reserved for padding",
        "split_path": "data/processed/splits.json", "split_sha256": sha(ROOT / "data/processed/splits.json"),
        "checkpoint_reproduction": reproduction,
    }
    (OUT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2) + "\n")

    cohort, variant_rows, failures = [], [], []
    test_rows = frame.iloc[splits["test"]]
    for row_index, row in test_rows.iterrows():
        source = str(row.original_representation)
        if len(tokenize(source)) > 256: continue
        variants, failed = generate_equivalents(source, seed=SEED + int(row_index), target=args.variants)
        valid = [v for v in variants if len(tokenize(v.variant_representation)) <= 256]
        if not valid: continue
        source_id = str(row.polymer_id)
        cohort.append({"source_id": source_id, "representation": source,
                       "ground_truth": float(row.property_value), "split": "test",
                       "dataset_row_index": int(row_index), "seed": SEED + int(row_index)})
        for item in failed: failures.append({"source_id": source_id, **item})
        for j, v in enumerate(valid):
            decision = is_equivalent(source, v.variant_representation)
            ts, tv = tokenize(source), tokenize(v.variant_representation)
            variant_rows.append({"source_id": source_id, "variant_id": f"{source_id}_v{j:03d}",
                "family": v.family, "equivalence_subset": v.subset,
                "source_representation": source, "variant_representation": v.variant_representation,
                "equivalent_pass": decision.equivalent, "validation_reason": decision.reason,
                "token_count_source": len(ts), "token_count_variant": len(tv),
                "sequence_length_source": len(ts), "sequence_length_variant": len(tv),
                "character_length_source": len(source), "character_length_variant": len(v.variant_representation),
                "special_token_position_changes": 0, "start_point_shift": int(v.family == 'start_point_shift'),
                "repeat_factor": 1, "orientation_flag": int(v.family == 'orientation_reversal'),
                "seed": v.generation_seed})
        if len(cohort) >= args.sources: break
    (OUT / "cohort_manifest.json").write_text(json.dumps({"seed": SEED, "sources": cohort}, indent=2) + "\n")
    pd.DataFrame(failures).to_csv(OUT / "failed_generation_attempts.csv", index=False)
    variants = pd.DataFrame(variant_rows); variants.to_csv(OUT / "equivalent_variants.csv", index=False)

    cohort_by_id = {r["source_id"]: r for r in cohort}; prediction_rows = []
    for source_id, group in variants.groupby("source_id", sort=False):
        texts = [group.iloc[0].source_representation] + group.variant_representation.tolist()
        values = predictor.predict(texts); source_pred = float(values[0]); truth = cohort_by_id[source_id]["ground_truth"]
        for (_, row), value in zip(group.iterrows(), values[1:]):
            prediction_rows.append({"source_id": source_id, "variant_id": row.variant_id, "family": row.family,
                "equivalence_subset": row.equivalence_subset, "prediction_source": source_pred,
                "prediction_variant": float(value), "signed_drift": float(value-source_pred),
                "absolute_drift": float(abs(value-source_pred)), "ground_truth": truth,
                "clean_error": float(abs(source_pred-truth))})
    predictions = pd.DataFrame(prediction_rows); predictions.to_csv(OUT / "predictions.csv", index=False)
    merged = predictions.merge(variants, on=["source_id","variant_id","family","equivalence_subset"])
    merged["token_count_delta"] = merged.token_count_variant - merged.token_count_source
    merged["character_length_delta"] = merged.character_length_variant - merged.character_length_source
    merged["tokenization_fragmentation"] = merged.token_count_variant / merged.character_length_variant.clip(lower=1)

    source_rows=[]
    for sid,g in predictions.groupby("source_id"):
        all_pred=np.r_[g.prediction_source.iloc[0],g.prediction_variant.to_numpy()]; center=all_pred.mean(); d=g.absolute_drift
        source_rows.append({"source_id":sid,"clean_absolute_error":g.clean_error.iloc[0],
            "Delta_equiv_original":d.max(),"Delta_equiv_centered":np.max(np.abs(all_pred-center)),
            "mean_absolute_drift":d.mean(),"median_absolute_drift":d.median(),"std_prediction":all_pred.std(),
            "prediction_range":all_pred.max()-all_pred.min(),"P90_absolute_drift":d.quantile(.90),
            "P95_absolute_drift":d.quantile(.95),"num_verified_variants":len(g)})
    sources=pd.DataFrame(source_rows); sources.to_csv(OUT/"source_robustness.csv",index=False)
    corr=[]
    for metric in ["Delta_equiv_original","Delta_equiv_centered"]:
        rho,lo,hi=bootstrap_rho(sources.clean_absolute_error,sources[metric])
        corr.append({"disagreement_metric":metric,"spearman_rho":rho,"bootstrap_ci_low":lo,"bootstrap_ci_high":hi,
                     "n_sources":len(sources),"classification_threshold":"NOT_RUN_no_preexisting_high_error_threshold"})
    pd.DataFrame(corr).to_csv(OUT/"error_vs_invariance.csv",index=False)
    pd.DataFrame(corr).to_csv(OUT/"reliability_analysis.csv",index=False)

    fam=[]
    for name,g in predictions.groupby("family"):
        worst=g.groupby("source_id").absolute_drift.max()
        fam.append({"family":name,"n_sources":g.source_id.nunique(),"n_variants":len(g),"mean_drift":g.absolute_drift.mean(),
            "median_drift":g.absolute_drift.median(),"P90":g.absolute_drift.quantile(.9),"P95":g.absolute_drift.quantile(.95),
            "max":g.absolute_drift.max(),"mean_source_level_worst_case_drift":worst.mean()})
    pd.DataFrame(fam).to_csv(OUT/"family_results.csv",index=False)
    subset=[]
    for name,g in predictions.groupby("equivalence_subset"):
        subset.append({"subset":name,"n_sources":g.source_id.nunique(),"n_variants":len(g),"mean_drift":g.absolute_drift.mean(),
            "median_drift":g.absolute_drift.median(),"P90":g.absolute_drift.quantile(.9),"P95":g.absolute_drift.quantile(.95),
            "max_drift":g.absolute_drift.max(),"mean_source_worst_case":g.groupby('source_id').absolute_drift.max().mean()})
    pd.DataFrame(subset).to_csv(OUT/"repeat_shift_vs_broader.csv",index=False)

    shortcut=[]
    for feature in ["token_count_delta","character_length_delta","tokenization_fragmentation","start_point_shift","orientation_flag"]:
        rho,p=spearmanr(merged[feature],merged.absolute_drift)
        shortcut.append({"analysis":"spearman","feature":feature,"estimate":rho,"p_value":p,"n_variants":len(merged)})
    X=pd.get_dummies(merged[["token_count_delta","character_length_delta","start_point_shift","orientation_flag","family"]],columns=["family"],drop_first=True).astype(float)
    design=np.column_stack([np.ones(len(X)),X.to_numpy()]); beta=np.linalg.lstsq(design,merged.absolute_drift.to_numpy(),rcond=None)[0]
    for name,value in zip(["intercept",*X.columns],beta): shortcut.append({"analysis":"OLS_drift_controlling_for_length","feature":name,"estimate":value,"p_value":np.nan,"n_variants":len(merged)})
    pd.DataFrame(shortcut).to_csv(OUT/"shortcut_analysis.csv",index=False)

    graph_path=ROOT/"results/phase11c_canonical_verification/1789326819/representation_invariance.csv"
    graph=pd.read_csv(graph_path); graph_row=graph[graph.Model.str.contains("GraphMPNN")].iloc[0]
    comparison=pd.DataFrame([{"model":"GraphMPNN control","model_type":"GraphMPNN","clean_metric":0.411,
        "mean_equiv_drift":graph_row.mean_drift,"median_equiv_drift":graph_row.median_drift,
        "P95_equiv_drift":graph_row.p95,"max_equiv_drift":graph_row["max"]},
        {"model":"Frozen sequence model","model_type":"Transformer","clean_metric":0.4619162976741791,
        "mean_equiv_drift":predictions.absolute_drift.mean(),"median_equiv_drift":predictions.absolute_drift.median(),
        "P95_equiv_drift":predictions.absolute_drift.quantile(.95),"max_equiv_drift":predictions.absolute_drift.max()}])
    comparison.to_csv(OUT/"graph_vs_sequence.csv",index=False)
    predictions.sort_values("absolute_drift",ascending=False).head(20).to_csv(OUT/"top_failures.csv",index=False)

    plt.hist(predictions.absolute_drift,bins=35); plt.xlabel("absolute drift (eV)"); savefig("sequence_equiv_drift_distribution.png")
    plt.hist(sources.Delta_equiv_original,bins=30); plt.xlabel("source worst-case drift (eV)"); savefig("sequence_worst_case_drift.png")
    plt.scatter(sources.clean_absolute_error,sources.Delta_equiv_original,s=12); plt.xlabel("clean absolute error"); plt.ylabel("worst drift"); savefig("sequence_clean_error_vs_equiv_drift.png")
    plt.scatter(sources.Delta_equiv_centered,sources.clean_absolute_error,s=12); plt.xlabel("centered disagreement"); plt.ylabel("clean absolute error"); savefig("sequence_disagreement_vs_error.png")
    predictions.boxplot(column="absolute_drift",by="family",rot=25); plt.suptitle(""); savefig("sequence_drift_by_family.png")
    predictions.boxplot(column="absolute_drift",by="equivalence_subset",rot=15); plt.suptitle(""); savefig("repeat_shift_vs_broader.png")
    plt.scatter(merged.token_count_delta,merged.absolute_drift,s=8); plt.xlabel("token-count delta"); plt.ylabel("absolute drift"); savefig("sequence_drift_vs_token_delta.png")
    comparison.plot.bar(x="model",y="P95_equiv_drift",legend=False); plt.ylabel("P95 drift (eV)"); savefig("graph_vs_sequence_invariance.png")
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    print(json.dumps({"reproduction":reproduction,"sources":len(sources),"variants":len(predictions),
        "mean":predictions.absolute_drift.mean(),"median":predictions.absolute_drift.median(),
        "p90":predictions.absolute_drift.quantile(.9),"p95":predictions.absolute_drift.quantile(.95),
        "max":predictions.absolute_drift.max()},indent=2))

if __name__ == "__main__": main()
