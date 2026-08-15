"""Phase 2B: clean evaluation of every model on the identical val/test samples.

Each model is scored with ITS OWN saved scaler, because that scaler is part of
the trained model -- it defines the mapping from network output back to Kelvin.
Using a different scaler at inference than at training would misreport the model.

That is exactly why the Phase 2A comparison was confounded: its scaler differed
from Phase 1's, so "same model, different scaler" and "different model" were not
separable. Phase 2B fixes the scaler at TRAINING time instead.

Also reports a scaler-swap diagnostic on the Phase 2A model: what it would score
if its outputs were mapped back through the Phase 1 scaler. That isolates how
much of the 2A gap is pure normalization arithmetic.
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

MODELS = {
    "phase1_baseline": "results/models/transformer_regressor",
    "phase2a_defended": "results/models/transformer_defended",
    "phase2b_controlled": "results/models/transformer_defended_phase2b",
    "phase2b_balanced": "results/models/transformer_defended_phase2b_balanced",
}


def encode(psmiles: str, vocab: list[str], max_len: int):
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    ids = [char2idx.get(t, 0) for t in tokenize(psmiles)]
    mask = [False] * len(ids)
    while len(ids) < max_len:
        ids.append(0)
        mask.append(True)
    return ids[:max_len], mask[:max_len]


def load_model(path: Path, vocab, cfg):
    arch = cfg["architecture"]
    model = TransformerRegressorModel(
        vocab_size=len(vocab),
        d_model=arch["d_model"],
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"],
        max_seq_len=arch["max_seq_len"],
        pooling=arch["pooling"],
    )
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    model.eval()
    return model


def predict(model, df, rep_col, vocab, max_len, scaler) -> np.ndarray:
    ids, masks = [], []
    for s in df[rep_col]:
        i, m = encode(s, vocab, max_len)
        ids.append(i)
        masks.append(m)
    with torch.no_grad():
        out = model(torch.tensor(ids), padding_mask=torch.tensor(masks))
    return np.asarray(scaler.inverse_transform(out.numpy()))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum(err**2))
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def main() -> None:
    data_cfg = load_config("configs/dataset.yaml")
    cfg = load_config("configs/model.yaml")
    rep_col, tgt_col = data_cfg["representation_column"], data_cfg["target_column"]
    proc = Path(data_cfg["processed_dir"])

    df = pd.read_csv(proc / "processed.csv")
    splits = json.loads((proc / "splits.json").read_text())
    vocab = json.loads((proc / "vocab.json").read_text())
    max_len = cfg["architecture"]["max_seq_len"]

    val_df, test_df = df.iloc[splits["val"]], df.iloc[splits["test"]]
    results: dict[str, dict] = {}

    for name, path_str in MODELS.items():
        path = Path(path_str)
        if not (path / "model.pt").exists():
            results[name] = {"status": "missing"}
            continue

        scaler = TargetScaler.load(path / "scaler.json")
        model = load_model(path, vocab, cfg)

        results[name] = {
            "scaler": {"mean": scaler.mean, "std": scaler.std},
            "val": metrics(
                val_df[tgt_col].to_numpy(float),
                predict(model, val_df, rep_col, vocab, max_len, scaler),
            ),
            "test": metrics(
                test_df[tgt_col].to_numpy(float),
                predict(model, test_df, rep_col, vocab, max_len, scaler),
            ),
        }

    # Diagnostic: Phase 2A model decoded through the Phase 1 scaler. Isolates how
    # much of the 2A result is normalization arithmetic rather than learning.
    p2a, p1 = Path(MODELS["phase2a_defended"]), Path(MODELS["phase1_baseline"])
    if (p2a / "model.pt").exists():
        model = load_model(p2a, vocab, cfg)
        s1 = TargetScaler.load(p1 / "scaler.json")
        results["phase2a_decoded_with_phase1_scaler"] = {
            "note": "DIAGNOSTIC ONLY -- mismatched scaler, not a valid model score",
            "val": metrics(
                val_df[tgt_col].to_numpy(float),
                predict(model, val_df, rep_col, vocab, max_len, s1),
            ),
            "test": metrics(
                test_df[tgt_col].to_numpy(float),
                predict(model, test_df, rep_col, vocab, max_len, s1),
            ),
        }

    out = Path("results/phase2b_clean_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"{'model':<42} {'split':<6} {'n':>3} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 82)
    for name, r in results.items():
        if r.get("status") == "missing":
            print(f"{name:<42} MISSING")
            continue
        for split in ("val", "test"):
            m = r[split]
            print(
                f"{name:<42} {split:<6} {m['n']:>3} {m['mae']:>8.2f} "
                f"{m['rmse']:>8.2f} {m['r2']:>8.2f}"
            )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
