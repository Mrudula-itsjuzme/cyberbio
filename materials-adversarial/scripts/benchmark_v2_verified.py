"""Verified Framework V2 benchmark.

Repairs the developmental benchmark (docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md) and produces
the only Framework V2 numbers that may be cited.

Guarantees enforced by construction:

* the ONLY predictor is :class:`materials_adv.framework.predictors.GraphMPNNPredictor`,
  hash-validated against RELEASE_MANIFEST.json (a stub cannot be reached: it is not
  imported here at all);
* sources come from the canonical validation split, stratified, with 0 train and 0 test
  rows, recorded with seed + split hash + manifest hash;
* every model call goes through
  :class:`materials_adv.framework.accounting.AttackEvaluator`, so
  ``predictor_calls == 1 + queries_used`` and ``queries_used <= Q`` are checked after
  every single run;
* the source is scored once, before the attack, and is not charged to ``Q``;
* preflight gates STOP the run if the model no longer reproduces the canonical clean
  metrics or if one-edit drift lands in the implausible 5-10 eV regime.

Outputs (numbers are written to CSV/JSON first and the markdown tables are then
generated FROM those artifacts -- no hand-typed values):

    results/framework_v2/verified_run_1/
    docs/FRAMEWORK_V2_ATTACK_RESULTS.md
    docs/SEARCH_COMPARISON_TABLE.md
    docs/ATTACK_COMPARISON_TABLE.md
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials_adv.data.scaler import TargetScaler  # noqa: E402
from materials_adv.data.tokenizer import tokenize  # noqa: E402
from materials_adv.domain.chemistry.attacks.aliphatic_carbon_substitution import (  # noqa: E402
    AliphaticCarbonSubstitutionAttack,
)
from materials_adv.domain.chemistry.attacks.scaffold_preserving import (  # noqa: E402
    ScaffoldPolicy,
    ScaffoldPreservingAttack,
)
from materials_adv.domain.chemistry.attacks.simple_substitution import (  # noqa: E402
    SimpleSubstitutionAttack,
)
from materials_adv.domain.chemistry.graph_metrics import count_wildcards, graph_change  # noqa: E402
from materials_adv.domain.chemistry.identity import CanonicalSmilesIdentity  # noqa: E402
from materials_adv.domain.chemistry.validator import RDKitValidityChecker  # noqa: E402
from materials_adv.framework.accounting import AttackEvaluator  # noqa: E402
from materials_adv.framework.budget import Budget  # noqa: E402
from materials_adv.framework.interfaces import Candidate  # noqa: E402
from materials_adv.framework.objectives import (  # noqa: E402
    TargetDecrease,
    TargetIncrease,
    UntargetedDrift,
)
from materials_adv.framework.predictors import GraphMPNNPredictor  # noqa: E402
from materials_adv.framework.search import (  # noqa: E402
    EvolutionarySearch,
    GreedySearch,
    MetropolisSearch,
    RandomSearch,
)

CKPT = ROOT / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
SCALER = ROOT / "results/models/transformer_regressor/scaler.json"
VOCAB = ROOT / "data/processed/vocab.json"
TRANSFORMER_CKPT = ROOT / "results/models/transformer_regressor/model.pt"

OUT = ROOT / "results/framework_v2/verified_run_1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SOURCES = 30
QUERY_BUDGETS = (10, 20, 50)
EDIT_BUDGET = 3
POPULATION_SIZE = 5
ELITE_SIZE = 2

EXPECTED_CLEAN = {"MAE": 0.411190, "RMSE": 0.595369, "R2": 0.828639}
CLEAN_TOLERANCE = 1e-3
CANONICAL_BOUNDED_MAX_DRIFT = 3.19

STOPPED: list[str] = []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles


def scaffold_of(smiles: str) -> str:
    return ScaffoldPolicy().scaffold_smiles(smiles)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def load_predictor() -> GraphMPNNPredictor:
    predictor = GraphMPNNPredictor(CKPT, SCALER)
    predictor.validate_against_release_manifest(ROOT / "RELEASE_MANIFEST.json")
    if predictor.parameter_count() != 27585:
        raise SystemExit("STOP: GraphMPNN parameter count is not 27585")
    return predictor


def verify_clean_model(predictor: GraphMPNNPredictor, splits: dict, df: pd.DataFrame) -> dict:
    val_df = df.iloc[splits["val"]].copy().reset_index(drop=True)
    preds = np.asarray(predictor.predict_batch(val_df["original_representation"].tolist()))
    truth = val_df["property_value"].astype(float).to_numpy()
    err = preds - truth
    metrics = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1 - np.sum(err**2) / np.sum((truth - truth.mean()) ** 2)),
        "n": int(len(val_df)),
    }
    for key, expected in EXPECTED_CLEAN.items():
        if abs(metrics[key] - expected) > CLEAN_TOLERANCE:
            raise SystemExit(
                f"STOP: validation {key} = {metrics[key]:.6f}, expected {expected:.6f}. "
                "The pipeline no longer reproduces the canonical clean model; attack "
                "analysis must not proceed."
            )
    return metrics


def verify_predictor_matches_phase11c(predictor: GraphMPNNPredictor, splits: dict, df: pd.DataFrame) -> dict:
    """Compare against the Phase 11C evaluator path (same graph pipeline, independent code)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "p11c", ROOT / "scripts/run_phase11c_canonical_verification.py"
    )
    p11c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p11c)

    sample = df.iloc[splits["val"]].reset_index(drop=True).head(20)
    canonical_preds = p11c.eval_model_canonical(
        predictor.model, sample, None, predictor.scaler, predictor.device, is_graph=True
    )["Predictions"]
    wrapper_preds = predictor.predict_batch(sample["original_representation"].tolist())
    diff = np.abs(np.asarray(canonical_preds) - np.asarray(wrapper_preds))
    result = {"n": len(sample), "mean_abs_diff": float(diff.mean()), "max_abs_diff": float(diff.max())}
    if result["max_abs_diff"] > 1e-6:
        raise SystemExit(
            f"STOP: V2 predictor differs from the Phase 11C evaluator by {result['max_abs_diff']}"
        )
    return result


# ---------------------------------------------------------------------------
# Verified source set
# ---------------------------------------------------------------------------
def build_source_set(splits: dict, df: pd.DataFrame) -> pd.DataFrame:
    """30 validation polymers, stratified low/medium/high bandgap + scaffold diversity."""
    val = df.iloc[splits["val"]].copy().reset_index(drop=True)
    val["bandgap"] = val["property_value"].astype(float)
    val["scaffold"] = val["original_representation"].map(scaffold_of)
    val["atoms"] = val["original_representation"].map(lambda s: Chem.MolFromSmiles(s).GetNumAtoms())
    val["canonical"] = val["original_representation"].map(canonical)

    q1, q2 = val["bandgap"].quantile([1 / 3, 2 / 3])
    strata = {
        "low": val[val["bandgap"] <= q1],
        "medium": val[(val["bandgap"] > q1) & (val["bandgap"] <= q2)],
        "high": val[val["bandgap"] > q2],
    }
    per_stratum = N_SOURCES // len(strata)

    chosen = []
    for name, frame in strata.items():
        # round-robin over scaffold groups -> structural diversity within the stratum
        groups = {}
        for _, row in frame.sort_values("canonical").iterrows():
            key = row["scaffold"] or f"ACYCLIC:{row['atoms']}"
            row = row.copy()
            row["scaffold_group"] = key
            groups.setdefault(key, []).append(row)
        ordered = sorted(groups.items())
        rng = random.Random(SEED)
        picked, cursor = 0, 0
        while picked < per_stratum and any(groups[k] for k, _ in ordered):
            key, _ = ordered[cursor % len(ordered)]
            if groups[key]:
                row = groups[key].pop(0)
                chosen.append({**row.to_dict(), "stratum": name})
                picked += 1
            cursor += 1
            if cursor > 10_000:
                break
        _ = rng  # determinism comes from sorted ordering; no randomness needed

    selected = pd.DataFrame(chosen).head(N_SOURCES).reset_index(drop=True)
    selected = selected.rename(columns={"canonical": "smiles"})
    selected["source_id"] = selected.index
    return selected


def write_source_artifacts(selected: pd.DataFrame, splits: dict) -> dict:
    table = selected[["source_id", "smiles", "bandgap", "stratum", "scaffold",
                      "scaffold_group", "atoms"]].copy()
    table.to_csv(OUT / "benchmark_sources.csv", index=False)

    manifest = {
        "seed": SEED,
        "n_sources": int(len(table)),
        "split_counts": {
            "train_count": int(sum(1 for _ in splits["train"])),
            "validation_count": int(sum(1 for _ in splits["val"])),
            "test_count": int(sum(1 for _ in splits["test"])),
        },
        "source_split_membership": {"train": 0, "validation": int(len(table)), "test": 0},
        "splits_sha256": sha256_file(ROOT / "data/processed/splits.json"),
        "benchmark_sources_sha256": sha256_file(OUT / "benchmark_sources.csv"),
        "stratification": dict(Counter(table["stratum"])),
        "unique_murcko_scaffolds": int(table["scaffold"].nunique()),
        "unique_scaffold_groups": int(table["scaffold_group"].nunique()),
        "unique_atom_counts": int(table["atoms"].nunique()),
        "selection": (
            "validation split only; tertiles of the validation bandgap distribution; "
            "round-robin across Murcko-scaffold groups within each tertile; deterministic"
        ),
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# One-edit sanity gate
# ---------------------------------------------------------------------------
def one_edit_sanity(predictor: GraphMPNNPredictor, selected: pd.DataFrame, n_sources: int = 10) -> dict:
    rows = []
    operator = SimpleSubstitutionAttack()
    validator = RDKitValidityChecker(require_single_component=True)
    for _, source in selected.head(n_sources).iterrows():
        smiles = source["smiles"]
        source_prediction = predictor.predict(smiles)
        for child in operator.apply(Candidate(smiles, provenance=["source"])):
            if not validator.is_valid(child):
                continue
            prediction = predictor.predict(child.identifier)
            rows.append({
                "source_id": int(source["source_id"]),
                "source_smiles": smiles,
                "candidate_smiles": child.identifier,
                "drift_eV": abs(prediction - source_prediction),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "one_edit_sanity.csv", index=False)
    summary = {
        "n_sources": n_sources,
        "n_candidates": int(len(frame)),
        "mean_drift_eV": float(frame["drift_eV"].mean()),
        "median_drift_eV": float(frame["drift_eV"].median()),
        "p95_drift_eV": float(frame["drift_eV"].quantile(0.95)),
        "max_drift_eV": float(frame["drift_eV"].max()),
        "forensic_reference": {"mean": 0.213, "median": 0.104, "max": 1.547},
    }
    if summary["max_drift_eV"] > CANONICAL_BOUNDED_MAX_DRIFT or summary["mean_drift_eV"] > 2.0:
        raise SystemExit(
            "STOP: one-edit drift is in the implausible regime "
            f"(mean {summary['mean_drift_eV']:.3f} eV, max {summary['max_drift_eV']:.3f} eV). "
            "Re-audit the pipeline before running multi-edit searches."
        )
    return summary


# ---------------------------------------------------------------------------
# Search plumbing
# ---------------------------------------------------------------------------
def make_strategy(name: str, operator, evaluator: AttackEvaluator, validator, seed: int):
    if name == "random":
        return RandomSearch(operator, evaluator, validator, seed=seed)
    if name == "greedy":
        return GreedySearch(operator, evaluator, validator, seed=seed)
    if name == "metropolis":
        return MetropolisSearch(operator, evaluator, validator, seed=seed, temperature=0.1)
    if name == "evolutionary":
        return EvolutionarySearch(
            operator, evaluator, validator, seed=seed,
            population_size=POPULATION_SIZE, elite_size=ELITE_SIZE,
        )
    raise ValueError(f"unknown strategy {name}")


def make_validator(smiles: str) -> RDKitValidityChecker:
    """Validity gate: parse + sanitise + single component + attachment count preserved.

    The attachment count is precomputed once per source; passing the reference SMILES
    would re-parse it on every single proposal.
    """
    return RDKitValidityChecker(
        require_single_component=True,
        require_attachment_count=count_wildcards(smiles),
    )


def make_operator(name: str, smiles: str, validator: RDKitValidityChecker):
    # The operators do not pre-filter: the search-layer validator is the authoritative
    # gate, and rejecting a proposal there is what makes invalid-proposal counts visible.
    base = SimpleSubstitutionAttack(validator=validator, validate_children=False)
    if name == "simple_substitution":
        return base
    if name == "aliphatic_carbon_substitution":
        return AliphaticCarbonSubstitutionAttack(validator=validator, validate_children=False)
    if name == "scaffold_preserving":
        return ScaffoldPreservingAttack(base, validator=validator)
    raise ValueError(f"unknown operator {name}")


def run_one(predictor, smiles: str, operator_name: str, strategy_name: str, q: int,
            objective, identity, seed: int = SEED) -> dict:
    source = Candidate(smiles, provenance=["source"])
    budget = Budget(q, EDIT_BUDGET)
    evaluator = AttackEvaluator(predictor, objective, budget, source, identity)
    validator = make_validator(smiles)
    operator = make_operator(operator_name, smiles, validator)
    search = make_strategy(strategy_name, operator, evaluator, validator, seed)
    outcome = search.search(source)
    invariants = evaluator.invariants()
    if not invariants["one_model_call_per_query"]:
        raise SystemExit(f"STOP: query accounting broke for {strategy_name}: {invariants}")
    if outcome.queries_used > q:
        raise SystemExit(f"STOP: {strategy_name} used {outcome.queries_used} > Q={q}")

    change = graph_change(smiles, outcome.best_representation)
    row = outcome.to_row()
    row.update({
        "operator": operator_name,
        "objective": type(objective).__name__,
        "source_smiles": smiles,
        "best_smiles": outcome.best_representation,
        "predictor_calls": invariants["predictor_calls"],
        "one_model_call_per_query": invariants["one_model_call_per_query"],
        "atom_edit_count": change.atom_edit_count,
        "structural_change_proxy": change.structural_change_proxy,
        "element_change_count": change.element_change_count,
        "atom_count_delta": change.atom_count_delta,
        "component_count_delta": change.component_count_delta,
        "best_is_source": outcome.best_representation == smiles,
    })
    return row


def aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)
    out = grouped.agg(
        n_runs=("best_drift", "size"),
        eligible_sources=("source_smiles", "nunique"),
        valid_candidate_rate=("valid_candidate_rate", "mean"),
        unique_candidate_rate=("unique_candidate_rate", "mean"),
        mean_drift=("best_drift", "mean"),
        median_drift=("best_drift", "median"),
        p90_drift=("best_drift", lambda s: float(np.percentile(s, 90))),
        p95_drift=("best_drift", lambda s: float(np.percentile(s, 95))),
        max_drift=("best_drift", "max"),
        mean_queries_used=("queries_used", "mean"),
        max_queries_used=("queries_used", "max"),
        duplicate_count=("duplicates", "sum"),
        invalid_proposal_count=("invalid_proposals", "sum"),
        mean_operator_edits=("best_operator_edits", "mean"),
        max_operator_edits=("best_operator_edits", "max"),
        mean_atom_edits=("atom_edit_count", "mean"),
        max_atom_edits=("atom_edit_count", "max"),
        mean_structural_change=("structural_change_proxy", "mean"),
        query_efficiency=("drift_per_query", "mean"),
        unanswered_runs=("best_is_source", "sum"),
    ).reset_index()
    return out


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def benchmark_strategies(predictor, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = CanonicalSmilesIdentity()
    rows = []
    for strategy in ("random", "greedy", "metropolis", "evolutionary"):
        for q in QUERY_BUDGETS:
            for _, source in selected.iterrows():
                row = run_one(predictor, source["smiles"], "simple_substitution", strategy, q,
                              UntargetedDrift(), identity)
                row["source_id"] = int(source["source_id"])
                row["stratum"] = source["stratum"]
                rows.append(row)
            print(f"   strategy={strategy} Q={q} done ({len(rows)} runs)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "strategy_runs.csv", index=False)
    summary = aggregate(frame, ["strategy", "query_budget"])
    summary.to_csv(OUT / "strategy_summary.csv", index=False)
    return frame, summary


def benchmark_targeted(predictor, selected: pd.DataFrame, q: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = CanonicalSmilesIdentity()
    rows = []
    for objective_name, objective in (("TARGET_INCREASE", TargetIncrease()),
                                      ("TARGET_DECREASE", TargetDecrease())):
        for strategy in ("random", "greedy", "metropolis", "evolutionary"):
            for _, source in selected.iterrows():
                row = run_one(predictor, source["smiles"], "simple_substitution", strategy, q,
                              objective, identity)
                row["source_id"] = int(source["source_id"])
                row["objective_name"] = objective_name
                row["signed_change_eV"] = row["best_prediction"] - row["source_prediction"]
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "targeted_runs.csv", index=False)

    summary_rows = []
    for (objective_name, strategy), part in frame.groupby(["objective_name", "strategy"]):
        signed = part["signed_change_eV"]
        desired = (signed > 0) if objective_name == "TARGET_INCREASE" else (signed < 0)
        summary_rows.append({
            "objective": objective_name,
            "strategy": strategy,
            "query_budget": q,
            "n_sources": int(len(part)),
            "directional_success_rate": float(desired.mean()),
            "mean_signed_change_eV": float(signed.mean()),
            "median_signed_change_eV": float(signed.median()),
            "p10_signed_change_eV": float(np.percentile(signed, 10)),
            "p90_signed_change_eV": float(np.percentile(signed, 90)),
            "mean_queries_used": float(part["queries_used"].mean()),
            "valid_candidate_rate": float(part["valid_candidate_rate"].mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "targeted_summary.csv", index=False)
    return frame, summary


def benchmark_operators(predictor, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = CanonicalSmilesIdentity()
    rows = []
    for operator in ("simple_substitution", "aliphatic_carbon_substitution", "scaffold_preserving"):
        for q in QUERY_BUDGETS:
            for _, source in selected.iterrows():
                row = run_one(predictor, source["smiles"], operator, "greedy", q,
                              UntargetedDrift(), identity)
                row["source_id"] = int(source["source_id"])
                rows.append(row)
            print(f"   operator={operator} Q={q} done ({len(rows)} runs)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "operator_runs.csv", index=False)

    summary_rows = []
    for (operator, q), part in frame.groupby(["operator", "query_budget"]):
        summary_rows.append({
            "operator": operator,
            "query_budget": q,
            "n_runs": int(len(part)),
            "valid_candidate_rate": float(part["valid_candidate_rate"].mean()),
            "unique_candidate_rate": float(part["unique_candidate_rate"].mean()),
            "mean_drift_eV": float(part["best_drift"].mean()),
            "median_drift_eV": float(part["best_drift"].median()),
            "max_drift_eV": float(part["best_drift"].max()),
            "mean_operator_edits": float(part["best_operator_edits"].mean()),
            "max_operator_edits": int(part["best_operator_edits"].max()),
            "mean_atom_edits": float(part["atom_edit_count"].mean()),
            "max_atom_edits": int(part["atom_edit_count"].max()),
            "mean_structural_change_proxy": float(part["structural_change_proxy"].mean()),
            "max_structural_change_proxy": int(part["structural_change_proxy"].max()),
            "query_efficiency_eV_per_query": float(part["drift_per_query"].mean()),
            "unanswered_runs": int(part["best_is_source"].sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "operator_summary.csv", index=False)
    return frame, summary


# ---------------------------------------------------------------------------
# Cross-model transfer
# ---------------------------------------------------------------------------
class TransformerOrdinaryPredictor:
    """Canonical ordinary Transformer (85,761 params) bound to the framework Predictor API."""

    def __init__(self, checkpoint=TRANSFORMER_CKPT, scaler_path=SCALER, vocab_path=VOCAB,
                 device=None, batch_size=32):
        from materials_adv.models.transformer import TransformerRegressorModel

        self.vocab = json.loads(Path(vocab_path).read_text())
        self.scaler = TargetScaler.load(scaler_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.checkpoint = Path(checkpoint)
        self.model = TransformerRegressorModel(
            vocab_size=len(self.vocab), d_model=64, n_layers=2, n_heads=4,
            dim_feedforward=128, dropout=0.1, max_seq_len=256, pooling="mean",
        )
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device,
                                            weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, smiles_list):
        char2idx = {c: i + 1 for i, c in enumerate(self.vocab)}
        max_len = 256
        out = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), self.batch_size):
                chunk = smiles_list[i:i + self.batch_size]
                ids, masks = [], []
                for smiles in chunk:
                    tokens = [char2idx.get(t, 0) for t in tokenize(smiles)]
                    mask = [False] * len(tokens)
                    while len(tokens) < max_len:
                        tokens.append(0)
                        mask.append(True)
                    ids.append(tokens[:max_len])
                    masks.append(mask[:max_len])
                y_hat = self.model(torch.tensor(ids, dtype=torch.long).to(self.device),
                                   padding_mask=torch.tensor(masks, dtype=torch.bool).to(self.device))
                scaled = y_hat.view(-1).cpu().numpy().reshape(-1, 1)
                out.extend(self.scaler.inverse_transform(scaled).flatten().tolist())
        return out

    def predict(self, representation):
        smiles = representation.identifier if isinstance(representation, Candidate) else str(representation)
        return float(self.predict_batch([smiles])[0])


def cross_model_transfer(predictor, strategy_frame: pd.DataFrame) -> dict:
    subset = strategy_frame[(strategy_frame["strategy"] == "greedy")
                            & (strategy_frame["query_budget"] == QUERY_BUDGETS[-1])]
    subset = subset.sort_values("source_id").head(N_SOURCES)
    transformer = TransformerOrdinaryPredictor()

    sources = subset["source_smiles"].tolist()
    candidates = subset["best_smiles"].tolist()
    gnn_src = np.asarray(predictor.predict_batch(sources))
    gnn_cand = np.asarray(predictor.predict_batch(candidates))
    tx_src = np.asarray(transformer.predict_batch(sources))
    tx_cand = np.asarray(transformer.predict_batch(candidates))

    gnn_delta = gnn_cand - gnn_src
    tx_delta = tx_cand - tx_src

    from scipy.stats import pearsonr, spearmanr

    def safe(corr):
        if len(set(gnn_delta.tolist())) < 2 or len(set(tx_delta.tolist())) < 2:
            return None
        return float(corr(gnn_delta, tx_delta)[0])

    mean_abs_gnn = float(np.mean(np.abs(gnn_delta)))
    mean_abs_tx = float(np.mean(np.abs(tx_delta)))
    rows = []
    for i, source_id in enumerate(subset["source_id"].tolist()):
        rows.append({
            "source_id": int(source_id),
            "source_smiles": sources[i],
            "candidate_smiles": candidates[i],
            "graphmpnn_source_eV": float(gnn_src[i]),
            "graphmpnn_candidate_eV": float(gnn_cand[i]),
            "graphmpnn_delta_eV": float(gnn_delta[i]),
            "transformer_source_eV": float(tx_src[i]),
            "transformer_candidate_eV": float(tx_cand[i]),
            "transformer_delta_eV": float(tx_delta[i]),
            "signed_agreement": bool(np.sign(gnn_delta[i]) == np.sign(tx_delta[i])),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "transfer_runs.csv", index=False)
    summary = {
        "n_candidates": int(len(rows)),
        "graphmpnn_checkpoint_sha256": sha256_file(CKPT),
        "transformer_checkpoint": str(TRANSFORMER_CKPT.relative_to(ROOT)),
        "transformer_checkpoint_sha256": sha256_file(TRANSFORMER_CKPT),
        "mean_abs_graphmpnn_delta_eV": mean_abs_gnn,
        "mean_abs_transformer_delta_eV": mean_abs_tx,
        "mean_graphmpnn_delta_eV": float(np.mean(gnn_delta)),
        "mean_transformer_delta_eV": float(np.mean(tx_delta)),
        "signed_agreement_rate": float(np.mean(np.sign(gnn_delta) == np.sign(tx_delta))),
        "pearson_r": safe(pearsonr),
        "spearman_r": safe(spearmanr),
        "transfer_ratio_transformer_over_graphmpnn": (
            mean_abs_tx / mean_abs_gnn if mean_abs_gnn > 0 else None
        ),
        "one_way_only": True,
        "note": "Candidates are chosen by the GraphMPNN and merely re-scored by the Transformer.",
    }
    (OUT / "transfer_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Phase 12B bridge
# ---------------------------------------------------------------------------
def phase12b_bridge(predictor, selected: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "p12b", ROOT / "scripts/run_phase12b_adaptive_search_audit.py"
    )
    p12b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p12b)

    class Canonical12BHarness:
        mutate_smiles = p12b.Phase12bEvaluator.mutate_smiles
        compute_edits = p12b.Phase12bEvaluator.compute_edits
        check_validity = p12b.Phase12bEvaluator.check_validity
        generate_valid_neighbor = p12b.Phase12bEvaluator.generate_valid_neighbor
        search_random = p12b.Phase12bEvaluator.search_random
        search_greedy = p12b.Phase12bEvaluator.search_greedy
        search_metropolis = p12b.Phase12bEvaluator.search_metropolis

        def __init__(self, adapter):
            self.predict_gnn = adapter.predict_batch
            self.predict_tx = None
            self.scaler = adapter.scaler
            from collections import defaultdict
            self.validity_stats = defaultdict(int)
            self.trajectories = []

    harness = Canonical12BHarness(predictor)
    identity = CanonicalSmilesIdentity()
    methods = {
        "random": harness.search_random,
        "greedy": harness.search_greedy,
        "metropolis": harness.search_metropolis,
    }

    rows = []
    for q in (10, 50):
        for method, canonical_fn in methods.items():
            for _, source in selected.iterrows():
                smiles = source["smiles"]
                _, drift_canonical, queries_canonical = canonical_fn("GraphMPNN", smiles, 0, q, SEED)

                v2 = run_one(predictor, smiles, "simple_substitution", method, q,
                             UntargetedDrift(), identity)
                rows.append({
                    "query_budget": q,
                    "method": method,
                    "source_id": int(source["source_id"]),
                    "source_smiles": smiles,
                    "phase12b_drift_eV": float(drift_canonical),
                    "phase12b_queries": int(queries_canonical),
                    "v2_drift_eV": float(v2["best_drift"]),
                    "v2_queries": int(v2["queries_used"]),
                    "v2_operator_edits": v2["best_operator_edits"],
                    "v2_atom_edits": v2["atom_edit_count"],
                    "drift_ratio_v2_over_phase12b": (
                        float(v2["best_drift"] / drift_canonical) if drift_canonical > 0 else None
                    ),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "phase12b_bridge.csv", index=False)
    summary = {
        "n_sources": int(selected.shape[0]),
        "query_budgets": [10, 50],
        "seed": SEED,
        "by_method": {},
    }
    for method, part in frame.groupby("method"):
        summary["by_method"][method] = {
            "phase12b_mean_drift_eV": float(part["phase12b_drift_eV"].mean()),
            "v2_mean_drift_eV": float(part["v2_drift_eV"].mean()),
            "phase12b_max_drift_eV": float(part["phase12b_drift_eV"].max()),
            "v2_max_drift_eV": float(part["v2_drift_eV"].max()),
        }
    summary["residual_differences_explained_by"] = [
        "operator semantics: Phase 12B mutates aliphatic uppercase element TOKENS "
        "(aromatic/charged/bracket atoms immutable, 11 allowed tokens); V2 Simple "
        "Substitution re-types any heavy atom including aromatic and attachment-adjacent "
        "ones (5 allowed elements)",
        "proposal space: Phase 12B restricts candidates to token-edit-distance <= 3 from "
        "the ORIGINAL source and rejects chemical no-ops; V2 restricts to <= 3 operator "
        "applications and dedupes canonically",
        "search implementation: Phase 12B greedy scores a fixed 5-neighbour batch and "
        "Phase 12B metropolis uses a fixed temperature of 0.1 in eV, but neither charges "
        "the source prediction; V2 recomputes neighbourhoods per step and dedupes before "
        "spending a query",
        "RNG stream: identical seed value, different random consumption order",
    ]
    (OUT / "phase12b_bridge_summary.json").write_text(json.dumps(summary, indent=2))
    return frame, summary


# ---------------------------------------------------------------------------
# Documentation generated FROM artifacts
# ---------------------------------------------------------------------------
def _md_table(frame: pd.DataFrame, columns: dict[str, str], floatfmt: str = "{:.3f}") -> str:
    def fmt(value):
        if isinstance(value, float):
            return floatfmt.format(value)
        return str(value)

    header = "| " + " | ".join(columns.values()) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[key]) for key in columns) + " |")
    return "\n".join(lines)


def write_docs(manifest, clean, equivalence, one_edit, strategy_summary, targeted_summary,
               operator_summary, transfer, bridge, accounting) -> None:
    developmental = pd.read_csv(
        ROOT / "results/framework_v2/developmental_run_1/attack_results.csv"
    ).head(18)

    attack_doc = f"""# Framework V2 Attack Results

Two runs live in this file. Only the second one may be cited.

## 1. Developmental run — `DEVELOPMENTAL_UNVERIFIED`, DEPRECATED

Verdict of the forensic audit: **MULTIPLE_PROTOCOL_ERRORS**
(`docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md`). These numbers are invalid and must not be
quoted: the predictor was `float(hash(smiles) % 1000) / 100.0`, 25/30 sources came from
**train** and 5/30 from **test**, and 482/540 runs exceeded their declared query budget.

Frozen artifact: `results/framework_v2/developmental_run_1/attack_results.csv`.

{_md_table(developmental, {"Strategy": "Strategy", "Operator": "Operator", "Q": "Q",
                           "Mean Drift": "Mean Drift (invalid)", "Max Drift": "Max Drift (invalid)"})}

## 2. Verified run 1 — `verified_run_1`

Validation-only 30-source stratified subset, canonical GraphMPNN
(`{manifest['benchmark_sources_sha256'][:16]}…` source manifest), every model call
charged through one budgeted scoring path.

* split membership: train {manifest['source_split_membership']['train']},
  validation {manifest['source_split_membership']['validation']},
  test {manifest['source_split_membership']['test']}
* splits sha256 `{manifest['splits_sha256']}`
* stratification {manifest['stratification']}, unique Murcko scaffolds
  {manifest['unique_murcko_scaffolds']}, unique scaffold groups
  {manifest['unique_scaffold_groups']}, unique atom counts {manifest['unique_atom_counts']}

### Clean model verification (preflight)

| metric | measured | canonical | within tolerance |
|---|---|---|---|
| MAE (eV) | {clean['MAE']:.6f} | 0.411190 | yes |
| RMSE (eV) | {clean['RMSE']:.6f} | 0.595369 | yes |
| R² | {clean['R2']:.6f} | 0.828639 | yes |
| max abs diff vs Phase 11C evaluator (20 samples) | {equivalence['max_abs_diff']:.1e} | 0 | yes |

### Query accounting

Every run satisfies `predictor_calls == 1 + queries_used` and `queries_used <= Q`
(source scored once, uncharged). Violations: **{accounting['accounting_violations']}**
across {accounting['total_runs']} runs.

### Untargeted search strategies (operator: SimpleSubstitution, 3-edit budget)

{_md_table(strategy_summary, {
        "strategy": "Strategy", "query_budget": "Q", "n_runs": "Runs",
        "mean_drift": "Mean drift (eV)", "median_drift": "Median (eV)",
        "p95_drift": "p95 (eV)", "max_drift": "Max (eV)",
        "mean_queries_used": "Mean queries", "unique_candidate_rate": "Unique cand. rate",
        "max_atom_edits": "Max atom edits"})}

### Targeted objectives (Q={targeted_summary['query_budget'].iloc[0]}) — DIRECTIONAL_SUCCESS_RATE

`DIRECTIONAL_SUCCESS_RATE` is the fraction of sources whose prediction moved in the
requested direction. It is a sign test, not attack success.

{_md_table(targeted_summary, {
        "objective": "Objective", "strategy": "Strategy",
        "directional_success_rate": "Directional success rate",
        "mean_signed_change_eV": "Mean signed change (eV)",
        "median_signed_change_eV": "Median signed change (eV)",
        "p10_signed_change_eV": "p10 (eV)", "p90_signed_change_eV": "p90 (eV)",
        "mean_queries_used": "Mean queries"})}

### One-edit sanity gate

{_md_table(pd.DataFrame([one_edit]), {
        "n_sources": "Sources", "n_candidates": "Candidates", "mean_drift_eV": "Mean (eV)",
        "median_drift_eV": "Median (eV)", "p95_drift_eV": "p95 (eV)",
        "max_drift_eV": "Max (eV)"})}

Canonical bounded maximum reference: {CANONICAL_BOUNDED_MAX_DRIFT} eV.
Forensic reference regime: mean 0.213, median 0.104, max 1.547 eV.

### Cross-model transfer (one-way: GraphMPNN-chosen candidates re-scored by the Transformer)

| quantity | value |
|---|---|
| candidates | {transfer['n_candidates']} |
| mean abs GraphMPNN Δ (eV) | {transfer['mean_abs_graphmpnn_delta_eV']:.3f} |
| mean abs Transformer Δ (eV) | {transfer['mean_abs_transformer_delta_eV']:.3f} |
| signed agreement rate | {transfer['signed_agreement_rate']:.3f} |
| Pearson r | {transfer['pearson_r'] if transfer['pearson_r'] is None else round(transfer['pearson_r'], 3)} |
| Spearman r | {transfer['spearman_r'] if transfer['spearman_r'] is None else round(transfer['spearman_r'], 3)} |
| transfer ratio (Transformer / GraphMPNN) | {transfer['transfer_ratio_transformer_over_graphmpnn'] if transfer['transfer_ratio_transformer_over_graphmpnn'] is None else round(transfer['transfer_ratio_transformer_over_graphmpnn'], 3)} |

### Phase 12B bridge (Q ∈ {{10, 50}}, all {bridge['n_sources']} verified sources)

{_md_table(pd.DataFrame([{"method": m, **v} for m, v in bridge["by_method"].items()]), {
        "method": "Method",
        "phase12b_mean_drift_eV": "Phase 12B mean (eV)",
        "v2_mean_drift_eV": "V2 mean (eV)",
        "phase12b_max_drift_eV": "Phase 12B max (eV)",
        "v2_max_drift_eV": "V2 max (eV)"})}

Residual differences are expected and explained by operator semantics, proposal space,
search implementation and RNG order (see `phase12b_bridge_summary.json`). Direct
numerical equivalence is NOT claimed.

**Metric naming:** chemistry-changing numbers here are `PREDICTION DRIFT`, never
"adversarial error".
"""

    search_doc = f"""# Search Strategy Comparison

Generated from `results/framework_v2/verified_run_1/strategy_summary.csv`.
Operator fixed to SimpleSubstitution, edit budget 3, 30 validation-only sources,
canonical GraphMPNN, source prediction uncharged.

{_md_table(strategy_summary, {
        "strategy": "Strategy", "query_budget": "Q",
        "mean_drift": "Mean drift (eV)", "median_drift": "Median (eV)",
        "p95_drift": "p95 (eV)", "max_drift": "Max (eV)",
        "mean_queries_used": "Mean queries used", "query_efficiency": "Drift per query (eV)",
        "unique_candidate_rate": "Unique candidate rate",
        "duplicate_count": "Duplicates", "unanswered_runs": "Runs with no accepted edit"})}

## Reading notes

* All strategies spend at most `Q` candidate queries; the source is scored once outside
  the budget. Duplicates are dropped BEFORE a query is spent.
* `Runs with no accepted edit` counts sources where no valid candidate was found rather
  than a stronger one.
* Random samples the source neighbourhood without exploitation; Metropolis is a
  temperature-0.1 eV walk, not formal Metropolis-Hastings.
"""

    attack_table = f"""# Attack Family Comparison

Generated from `results/framework_v2/verified_run_1/`.

| Attack | Representation-preserving? | Chemistry-changing? | Edit unit | Validator | Budget | Oracle required? | Status |
|---|---|---|---|---|---|---|---|
| Equivalent SMILES | yes | no | none | RDKit canonical | n/a | no | canonical baseline |
| Random substitution | no | yes | 1 atom per application | RDKit + connectivity + attachment | 3 | no | canonical stress |
| SimpleSubstitution (V2) | no | yes | 1 atom per application | RDKit + connectivity + attachment | 3 | no | VERIFIED |
| AliphaticCarbonSubstitutionAttack | no | yes | 1 application = up to 4 atom edits | RDKit + connectivity + attachment | 3 | no | VERIFIED |
| Scaffold-preserving | no | yes | inherited from base operator | RDKit + Murcko subgraph retention | 3 | no | VERIFIED (subset) |
| True fragment/subgraph replacement | no | yes | fragment swap | not implemented | n/a | no | DESIGN ONLY |

Operator comparison (fixed strategy: greedy; from `operator_summary.csv`):

{_md_table(operator_summary, {
        "operator": "Operator", "query_budget": "Q",
        "mean_drift_eV": "Mean drift (eV)", "max_drift_eV": "Max (eV)",
        "mean_operator_edits": "Mean operator edits", "max_operator_edits": "Max operator edits",
        "max_atom_edits": "Max atom edits",
        "max_structural_change_proxy": "Max structural-change proxy",
        "query_efficiency_eV_per_query": "Drift per query (eV)",
        "unanswered_runs": "Runs with no accepted edit"})}

`AliphaticCarbonSubstitutionAttack` was previously misnamed
`FunctionalGroupReplacementAttack`; it substitutes aliphatic carbons and does not
manipulate functional-group fragments (audit section 10). Its operator-edit count is not
comparable to its atom-edit count, so both are reported.

Framework-V2 attack numbers are `DEVELOPMENTAL_UNVERIFIED` for the developmental run and
`VERIFIED` only for `verified_run_1`.
"""

    (ROOT / "docs/FRAMEWORK_V2_ATTACK_RESULTS.md").write_text(attack_doc)
    (ROOT / "docs/SEARCH_COMPARISON_TABLE.md").write_text(search_doc)
    (ROOT / "docs/ATTACK_COMPARISON_TABLE.md").write_text(attack_table)


# ---------------------------------------------------------------------------
def main() -> None:
    started = time.time()
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    df = pd.read_csv(ROOT / "data/processed/processed.csv")

    print("Preflight: loading canonical GraphMPNN ...", flush=True)
    predictor = load_predictor()

    print("Preflight: rebuilding clean validation metrics ...", flush=True)
    clean = verify_clean_model(predictor, splits, df)
    equivalence = verify_predictor_matches_phase11c(predictor, splits, df)
    print(f"   MAE={clean['MAE']:.6f} RMSE={clean['RMSE']:.6f} R2={clean['R2']:.6f} "
          f"| max |Δ| vs Phase 11C = {equivalence['max_abs_diff']:.1e}")

    print("Building verified validation-only source set ...", flush=True)
    selected = build_source_set(splits, df)
    manifest = write_source_artifacts(selected, splits)
    print(f"   {len(selected)} sources | strata {manifest['stratification']} "
          f"| unique Murcko scaffolds {manifest['unique_murcko_scaffolds']} "
          f"| unique scaffold groups {manifest['unique_scaffold_groups']}", flush=True)

    print("One-edit sanity gate ...", flush=True)
    one_edit = one_edit_sanity(predictor, selected)
    print(f"   mean {one_edit['mean_drift_eV']:.3f} median {one_edit['median_drift_eV']:.3f} "
          f"max {one_edit['max_drift_eV']:.3f} eV", flush=True)

    print("Untargeted search-strategy benchmark ...", flush=True)
    strategy_frame, strategy_summary = benchmark_strategies(predictor, selected)

    print("Targeted benchmark ...", flush=True)
    _, targeted_summary = benchmark_targeted(predictor, selected)

    print("Operator benchmark ...", flush=True)
    _, operator_summary = benchmark_operators(predictor, selected)

    print("Cross-model transfer ...", flush=True)
    transfer = cross_model_transfer(predictor, strategy_frame)

    print("Phase 12B bridge ...", flush=True)
    _, bridge = phase12b_bridge(predictor, selected)

    total_runs = int(len(strategy_frame) + len(pd.read_csv(OUT / "targeted_runs.csv"))
                     + len(pd.read_csv(OUT / "operator_runs.csv")))
    accounting = {
        "total_runs": total_runs,
        "accounting_violations": 0,
        "invariant": "predictor_calls == 1 + queries_used and queries_used <= Q",
        "source_prediction_charged_to_Q": False,
    }

    write_docs(manifest, clean, equivalence, one_edit, strategy_summary,
               targeted_summary, operator_summary, transfer, bridge, accounting)

    summary = {
        "run": "verified_run_1",
        "elapsed_seconds": round(time.time() - started, 1),
        "seed": SEED,
        "query_budgets": list(QUERY_BUDGETS),
        "edit_budget": EDIT_BUDGET,
        "clean_metrics": clean,
        "predictor_equivalence": equivalence,
        "one_edit_sanity": one_edit,
        "source_manifest": manifest,
        "accounting": accounting,
        "strategy_summary": strategy_summary.to_dict(orient="records"),
        "targeted_summary": targeted_summary.to_dict(orient="records"),
        "operator_summary": operator_summary.to_dict(orient="records"),
        "transfer": transfer,
        "phase12b_bridge": bridge,
    }
    (OUT / "verified_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nVerified benchmark complete in {summary['elapsed_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
