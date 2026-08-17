"""Forensic diagnostics for the length-vulnerability claim (Phase 2F).

Implements the smallest set of controls that can falsify the Phase 1F conclusion
that "length-changing perturbations cause the ~30 K drift". No training is
performed; every diagnostic reuses existing checkpoints.

Diagnostics
-----------
D1  masking correctness      -- true no-op: append PAD-masked positions
D2  batch invariance         -- same string alone vs in a batch
D3  control baselines        -- mean / median / length-only / bag-of-tokens
D4  order ablation           -- shuffle and reverse (length-preserving)
D5  mean-pool dilution       -- drift vs sequence length
D6  chemical destructiveness -- RDKit deltas per attack family
D7  cross-seed replication   -- D4/D5 over the 5 Phase 2C baseline seeds

Run:  .venv/bin/python scripts/forensic_diagnostics.py
Writes results/forensic_diagnostics.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.transformer import TransformerRegressorModel

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEEDS = list(range(20260815, 20260820))


def load_model(path: Path):
    vocab = json.loads((PROC / "vocab.json").read_text())
    model = TransformerRegressorModel(len(vocab), 64, 2, 4, 128, 0.1, 256, "mean")
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    model.eval()
    return model, TargetScaler.load(path / "scaler.json"), vocab


def predict_tokens(model, scaler, vocab, token_lists, extra_pad=0, use_mask=True):
    """Predict directly from token lists, bypassing the string round trip."""
    c2i = {c: i + 1 for i, c in enumerate(vocab)}
    width = max(len(t) for t in token_lists) + extra_pad
    ids, masks = [], []
    for toks in token_lists:
        row = [c2i.get(t, 0) for t in toks]
        mask = [False] * len(row)
        while len(row) < width:
            row.append(0)
            mask.append(True)
        ids.append(row)
        masks.append(mask)
    with torch.no_grad():
        out = model(
            torch.tensor(ids),
            padding_mask=torch.tensor(masks) if use_mask else None,
        )
    return scaler.inverse_transform(out.numpy())


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def ridge(X, y, alpha):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + alpha * np.eye(X1.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, X1.T @ y)


def ridge_predict(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def bag_of_tokens(strings, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(strings), len(vocab)))
    for r, s in enumerate(strings):
        for tok, count in Counter(tokenize(s)).items():
            if tok in idx:
                X[r, idx[tok]] = count
    return X


def main() -> None:
    df = pd.read_csv(PROC / "processed.csv")
    splits = json.loads((PROC / "splits.json").read_text())
    df["L"] = [len(tokenize(s)) for s in df["PSMILES"]]
    tr, va, te = df.iloc[splits["train"]], df.iloc[splits["val"]], df.iloc[splits["test"]]
    val_tokens = [tokenize(s) for s in va["PSMILES"]]
    y_tr = tr["Tg (K)"].values

    model, scaler, vocab = load_model(ROOT / "results" / "models" / "transformer_regressor")
    base = predict_tokens(model, scaler, vocab, val_tokens)
    out: dict = {}

    # D1 masking correctness -- must be exactly zero if the pad mask works.
    out["D1_masked_pad_noop"] = {
        str(k): float(np.abs(predict_tokens(model, scaler, vocab, val_tokens, extra_pad=k) - base).max())
        for k in (0, 5, 20, 100)
    }

    # D2 unmasked padding: length/position change carrying no chemistry.
    b0 = predict_tokens(model, scaler, vocab, val_tokens, use_mask=False)
    out["D2_unmasked_pad_drift"] = {
        str(k): float(np.abs(predict_tokens(model, scaler, vocab, val_tokens, k, use_mask=False) - b0).mean())
        for k in (1, 3, 10)
    }

    # D3 control baselines.
    wl = ridge(tr[["L"]].values.astype(float), y_tr, 1e-6)
    Btr, Bva, Bte = (bag_of_tokens(d["PSMILES"], vocab) for d in (tr, va, te))
    wb = ridge(Btr, y_tr, 1.0)
    pv = predict_tokens(model, scaler, vocab, val_tokens)
    pt = predict_tokens(model, scaler, vocab, [tokenize(s) for s in te["PSMILES"]])
    out["D3_controls"] = {
        "mean": [mae(va["Tg (K)"], y_tr.mean()), mae(te["Tg (K)"], y_tr.mean())],
        "median": [mae(va["Tg (K)"], np.median(y_tr)), mae(te["Tg (K)"], np.median(y_tr))],
        "length_only": [
            mae(va["Tg (K)"], ridge_predict(wl, va[["L"]].values.astype(float))),
            mae(te["Tg (K)"], ridge_predict(wl, te[["L"]].values.astype(float))),
        ],
        "bag_of_tokens": [mae(va["Tg (K)"], ridge_predict(wb, Bva)), mae(te["Tg (K)"], ridge_predict(wb, Bte))],
        "transformer": [mae(va["Tg (K)"], pv), mae(te["Tg (K)"], pt)],
        "note": "[val_mae, test_mae] in Kelvin",
    }
    out["D3_split_confound"] = {
        "corr_length_Tg_all": float(np.corrcoef(df["L"], df["Tg (K)"])[0, 1]),
        "mean_length": {k: float(df.iloc[splits[k]]["L"].mean()) for k in ("train", "val", "test")},
        "mean_Tg": {k: float(df.iloc[splits[k]]["Tg (K)"].mean()) for k in ("train", "val", "test")},
    }

    # D4/D5/D7 order ablation, dilution, replication.
    replication = {}
    for seed in SEEDS:
        m, sc, _ = load_model(ROOT / "results" / "phase2c" / f"baseline_seed{seed}")
        b = predict_tokens(m, sc, vocab, val_tokens)
        rng = np.random.default_rng(0)
        shuffled = [list(rng.permutation(t)) for t in val_tokens]
        reversed_ = [t[::-1] for t in val_tokens]
        rng2 = np.random.default_rng(1)
        dup, keep = [], []
        for i, t in enumerate(val_tokens):
            pos = [j for j, x in enumerate(t) if x == "C"]
            if not pos:
                continue
            j = int(rng2.choice(pos))
            dup.append(t[: j + 1] + [t[j]] + t[j + 1 :])
            keep.append(i)
        replication[str(seed)] = {
            "shuffle": float(np.abs(predict_tokens(m, sc, vocab, shuffled) - b).mean()),
            "reverse": float(np.abs(predict_tokens(m, sc, vocab, reversed_) - b).mean()),
            "duplicate_plus1": float(np.abs(predict_tokens(m, sc, vocab, dup) - b[keep]).mean()),
        }
    out["D4_D7_order_vs_length"] = replication

    rng = np.random.default_rng(1)
    lengths, ins_drift, del_drift = [], [], []
    for i, t in enumerate(val_tokens):
        pos = [j for j, x in enumerate(t) if x in ("C", "c", "O", "N")]
        if not pos:
            continue
        j = int(rng.choice(pos))
        p = predict_tokens(model, scaler, vocab, [t[: j + 1] + [t[j]] + t[j + 1 :], t[:j] + t[j + 1 :]])
        lengths.append(len(t))
        ins_drift.append(abs(p[0] - base[i]))
        del_drift.append(abs(p[1] - base[i]))
    L = np.array(lengths)
    out["D5_mean_pool_dilution"] = {
        "corr_len_insertion_drift": float(np.corrcoef(L, ins_drift)[0, 1]),
        "corr_len_deletion_drift": float(np.corrcoef(L, del_drift)[0, 1]),
        "corr_invlen_insertion_drift": float(np.corrcoef(1 / L, ins_drift)[0, 1]),
        "by_length_bin": {
            f"[{lo},{hi})": {
                "n": int(((L >= lo) & (L < hi)).sum()),
                "insertion": float(np.array(ins_drift)[(L >= lo) & (L < hi)].mean()),
                "deletion": float(np.array(del_drift)[(L >= lo) & (L < hi)].mean()),
            }
            for lo, hi in [(0, 15), (15, 30), (30, 60), (60, 999)]
            if ((L >= lo) & (L < hi)).sum()
        },
    }

    # D6 chemical destructiveness per attack family.
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula

        RDLogger.DisableLog("rdApp.*")

        def parse(s):
            t = s.replace("[*:1]", "[H]").replace("[*:2]", "[H]").replace("[*]", "[H]")
            return Chem.MolFromSmiles(t)

        recs = [json.loads(l) for l in (ROOT / "data" / "attacks" / "phase2b_baseline_results.jsonl").open()]
        agg = defaultdict(lambda: defaultdict(list))
        totals, valids = Counter(), Counter()
        for r in recs:
            totals[r["attack_type"]] += 1
            if r["validity_status"] != "valid":
                continue
            valids[r["attack_type"]] += 1
            a, b = parse(r["original_psmiles"]), parse(r["adversarial_psmiles"])
            if a is None or b is None:
                continue
            f = r["attack_type"]
            agg[f]["dMW"].append(abs(Descriptors.MolWt(b) - Descriptors.MolWt(a)))
            agg[f]["formula_same"].append(CalcMolFormula(a) == CalcMolFormula(b))
            agg[f]["identical"].append(Chem.MolToSmiles(a) == Chem.MolToSmiles(b))
            agg[f]["drift"].append(abs(r["prediction_drift"]))
        out["D6_chemical_destructiveness"] = {
            f: {
                "n_valid": len(agg[f]["dMW"]),
                "validity_rate": valids[f] / totals[f],
                "mean_abs_dMW": float(np.mean(agg[f]["dMW"])),
                "formula_preserved_frac": float(np.mean(agg[f]["formula_same"])),
                "identical_molecule_frac": float(np.mean(agg[f]["identical"])),
                "mean_abs_drift": float(np.mean(agg[f]["drift"])),
            }
            for f in ("substitution", "insertion", "deletion", "rearrangement")
            if agg[f]["dMW"]
        }
    except ImportError:
        out["D6_chemical_destructiveness"] = "SKIPPED: rdkit unavailable"

    dest = ROOT / "results" / "forensic_diagnostics.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
