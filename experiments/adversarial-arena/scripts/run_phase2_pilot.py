"""
Phase 2 pilot runner for probabilistic attackers.

Evaluates CEM, SMC, ValiditySMC against the Arena TransPolymer defender
using the same configuration as the frozen baseline pilot:
  - same seeds (20260920, 20260921, 20260922)
  - same query budget (25/source)
  - same round count (3)
  - same defender config (Adam, lr=0.001, 1 epoch, batch=16, top-k=4)

Each attacker is run independently per seed (no Thompson competition in Phase 2).
All three baseline held-out attackers (RandomSMILES, MCMC, Evolutionary) are
evaluated against each round's defender checkpoint for the cross-training matrix.

Usage:
  python scripts/run_phase2_pilot.py --attacker cem --seed 20260920
  python scripts/run_phase2_pilot.py --attacker smc --seed 20260920 --reset
  python scripts/run_phase2_pilot.py --attacker validity_smc --seed 20260920
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent.parent / "materials-adversarial" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


QUERY_BUDGET = 25
N_SOURCES = 10
N_ROUNDS = 3
HARD_EXAMPLE_K = 4
ATTACKER_CHOICES = ("cem", "smc", "validity_smc")

DATA_CSV = ROOT.parent.parent / "materials-adversarial" / "data" / "processed" / "processed.csv"


def _make_sample(row):
    from arena.contracts import SourceSample
    return SourceSample(
        source_id=str(row.polymer_id),
        input=str(row.original_representation),
        target=float(row.property_value),
    )


def _load_sources(seed: int, n: int) -> list:
    import pandas as pd
    df = pd.read_csv(DATA_CSV)
    val = df[df["split"] == "val"].head(n * 4).sample(n=min(n, len(df[df["split"] == "val"])), random_state=seed)
    return [_make_sample(r) for r in val.itertuples()]


def _load_clean_data(seed: int, n: int = 50) -> list:
    import pandas as pd
    df = pd.read_csv(DATA_CSV)
    train = df[df["split"] == "train"].head(n)
    return [_make_sample(r) for r in train.itertuples()]


def build_attacker(name: str, seed: int, carry_state: bool):
    from arena.probabilistic_attackers import cem_attacker, smc_attacker, validity_smc_attacker
    factories = {
        "cem": cem_attacker,
        "smc": smc_attacker,
        "validity_smc": validity_smc_attacker,
    }
    return factories[name](seed=seed, query_budget=QUERY_BUDGET, carry_state=carry_state)


def build_baseline_evaluators(seed: int) -> dict:
    """Build the three baseline static attackers for cross-training evaluation."""
    from arena.attackers import accounted_mcmc_adapter, evolutionary_adapter, random_smiles_adapter
    return {
        "RandomSMILES": random_smiles_adapter(seed=seed, proposal_budget=2),
        "MCMC": accounted_mcmc_adapter(seed=seed, query_budget=QUERY_BUDGET),
        "Evolutionary": evolutionary_adapter(seed=seed, query_budget=QUERY_BUDGET),
    }


def evaluate_attackers(attackers: dict, samples, defender) -> dict:
    """Evaluate all attackers against a defender checkpoint. Returns drift stats per attacker."""
    results = {}
    for name, att in attackers.items():
        try:
            records = att.propose(samples, defender)
            drifts = [r.drift for r in records if r.drift is not None]
            results[name] = {
                "mean_drift": sum(drifts) / len(drifts) if drifts else 0.0,
                "max_drift": max(drifts) if drifts else 0.0,
                "n_candidates": len(records),
            }
        except Exception as e:
            results[name] = {"mean_drift": None, "max_drift": None, "error": str(e)}
    return results


def run_phase2_pilot(attacker_name: str, seed: int, carry_state: bool, out_dir: Path):
    from arena.transpolymer import TransPolymerDefender

    out_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "attacker": attacker_name,
        "carry_state": carry_state,
        "seed": seed,
        "query_budget": QUERY_BUDGET,
        "n_sources": N_SOURCES,
        "n_rounds": N_ROUNDS,
        "status": "RUNNING",
        "rounds": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = out_dir / "run_summary.json"

    # Resume support: find last completed round
    start_round = 0
    for r in range(N_ROUNDS):
        if (out_dir / f"round_{r:03d}" / "COMPLETE.json").exists():
            start_round = r + 1

    if start_round > 0:
        print(f"Resuming from round {start_round}")
        run_summary = json.loads(summary_path.read_text()) if summary_path.exists() else run_summary

    # We need paths to canonical model
    checkpoint_dir = ROOT.parent.parent / "materials-adversarial" / "results" / "models" / "transformer_regressor"
    
    defender = TransPolymerDefender(
        project_root=ROOT.parent.parent / "materials-adversarial",
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        lambda_rep=0.5,
        learning_rate=0.001,
        epochs=1
    )
    if start_round > 0:
        ckpt = out_dir / f"round_{start_round - 1:03d}" / "defender_checkpoint.json"
        if ckpt.exists():
            defender.load_checkpoint(str(ckpt))

    sources = _load_sources(seed=seed, n=N_SOURCES)
    attacker = build_attacker(attacker_name, seed, carry_state)
    baseline_evaluators = build_baseline_evaluators(seed)

    # Restore attacker state if resuming carry-state run
    if carry_state and start_round > 0:
        last_state_path = out_dir / f"round_{start_round - 1:03d}" / f"{attacker_name}_state.json"
        if last_state_path.exists():
            saved = json.loads(last_state_path.read_text())
            attacker._round_id = saved.get("round_id", start_round)
            if attacker_name == "cem" and "probs" in saved:
                attacker._probs = saved["probs"]
            elif attacker_name in ("smc", "validity_smc"):
                if "particle_state" in saved:
                    attacker._particle_state = saved.get("particle_state")
                    attacker._weight_state = saved.get("weight_state")

    hard_examples = []

    # D_0 baseline evaluation
    if start_round == 0:
        print("Evaluating D_0 baseline...")
        d0_results = evaluate_attackers({attacker_name: attacker, **baseline_evaluators}, sources, defender)
        d0_dir = out_dir / "round_000"
        d0_dir.mkdir(exist_ok=True)
        (d0_dir / "baseline_cross_attack_matrix.csv").write_text(
            "attacker,mean_drift,max_drift,n_candidates\n" +
            "\n".join(f"{n},{v.get('mean_drift','')},{v.get('max_drift','')},{v.get('n_candidates','')}"
                      for n, v in d0_results.items())
        )
        # Reset attacker for actual Round 0
        attacker.reset(seed)

    for round_id in range(start_round, N_ROUNDS):
        round_dir = out_dir / f"round_{round_id:03d}"
        round_dir.mkdir(exist_ok=True)
        print(f"\n=== Round {round_id} | {attacker_name} | seed {seed} ===")

        # Snapshot state BEFORE this round's proposals (for G_adapt)
        state_before = dict(attacker.get_state())
        (round_dir / "attacker_state_before.json").write_text(json.dumps(state_before, indent=2))

        # Attack
        t_attack = time.monotonic()
        candidates = attacker.propose(sources, defender)
        attack_time = time.monotonic() - t_attack
        attacker_metrics = attacker.get_metrics()
        state_after = dict(attacker.get_state())

        (round_dir / "attacker_state_after.json").write_text(json.dumps(state_after, indent=2))
        (round_dir / "attacker_metrics.json").write_text(json.dumps(attacker_metrics, indent=2))

        # Save probabilistic state for resume
        prob_state = {
            "round_id": attacker._round_id,
            "carry_state": carry_state,
        }
        if attacker_name == "cem":
            prob_state["probs"] = attacker._probs
        elif attacker_name in ("smc", "validity_smc"):
            prob_state["particle_state"] = attacker._particle_state
            prob_state["weight_state"] = attacker._weight_state
        (round_dir / f"{attacker_name}_state.json").write_text(json.dumps(prob_state, indent=2))

        # Candidates
        import csv, io
        cand_lines = []
        drifts = []
        for c in candidates:
            if c.drift is not None:
                drifts.append(c.drift)
            cand_lines.append(json.dumps(c.to_dict()))
        (round_dir / "candidates.jsonl").write_text("\n".join(cand_lines))

        # Hard examples
        new_hard = sorted(candidates, key=lambda c: c.drift or 0, reverse=True)[:HARD_EXAMPLE_K]
        hard_examples.extend(new_hard)
        (round_dir / "hard_examples.jsonl").write_text("\n".join(json.dumps(h.to_dict()) for h in hard_examples))

        # Compute G_adapt(round_id): R(A_t+1, D_t) - R(A_t, D_t)
        # A_t = pre-update state, A_t+1 = post-update state
        # We approximate: drift difference is captured implicitly by state snapshots
        # For exact G_adapt we'd need to run both states against D_t — recorded for export
        mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
        drift_before = state_before.get("mean_drift")

        # Defender training
        print(f"  Training defender on {len(hard_examples)} hard examples...")
        defender_metrics_before = defender.evaluate(sources)
        clean_data = _load_clean_data(seed=seed, n=50)
        from arena.contracts import SourceSample as _SS
        clean_samples = [_SS(source_id=s.source_id, input=s.input, target=s.target) for s in clean_data]
        train_metrics = defender.train(clean_samples, hard_examples, round_id=round_id)
        defender_metrics_after = defender.evaluate(sources)
        ckpt_path = defender.checkpoint(str(round_dir / "defender_checkpoint.json"))

        # Cross-attack evaluation against updated D_{round_id+1}
        cross_results = evaluate_attackers({attacker_name: attacker, **baseline_evaluators}, sources, defender)
        cross_csv_lines = ["round,attacker,mean_drift,max_drift,n_candidates"]
        for att_name, r in cross_results.items():
            cross_csv_lines.append(f"{round_id},{att_name},{r.get('mean_drift','')},{r.get('max_drift','')},{r.get('n_candidates','')}")
        (round_dir / "cross_attack_matrix.csv").write_text("\n".join(cross_csv_lines))

        # Defender gain
        def_drift_before = defender_metrics_before.get("mean_robustness_drift") or attacker_metrics.get("mean_drift") or mean_drift
        def_drift_after = cross_results.get(attacker_name, {}).get("mean_drift") or mean_drift
        defender_gain = def_drift_before / max(def_drift_after, 1e-8) if def_drift_after else None

        round_summary = {
            "round": round_id,
            "attacker": attacker_name,
            "carry_state": carry_state,
            "seed": seed,
            "mean_drift": mean_drift,
            "max_drift": max(drifts) if drifts else 0.0,
            "drift_per_query": mean_drift / max(attacker_metrics.get("target_model_queries", 1), 1),
            "valid_count": attacker_metrics.get("valid_count", 0),
            "invalid_count": attacker_metrics.get("invalid_count", 0),
            "proposal_count": attacker_metrics.get("proposal_count", 0),
            "target_model_queries": attacker_metrics.get("target_model_queries", 0),
            "validity_rate": attacker_metrics.get("validity_rate"),
            "hard_example_buffer_size": len(hard_examples),
            "defender_clean_rmse": defender_metrics_after.get("clean_rmse"),
            "defender_gain": defender_gain,
            "attacker_state_before": state_before,
            "attacker_state_after": state_after,
            "g_adapt_inputs_available": drift_before is not None,
            "attack_runtime_seconds": attack_time,
        }

        # Attacker-specific diagnostics
        if attacker_name == "cem":
            round_summary["cem_entropy"] = state_after.get("entropy")
            round_summary["cem_probs"] = state_after.get("probs")
        elif attacker_name in ("smc", "validity_smc"):
            round_summary["smc_ess"] = state_after.get("ess")
            round_summary["smc_particle_count"] = state_after.get("particle_count")
        if attacker_name == "validity_smc":
            round_summary["validity_posterior"] = state_after.get("validity_posterior")

        (round_dir / "round_summary.json").write_text(json.dumps(round_summary, indent=2))
        (round_dir / "defender_metrics.json").write_text(json.dumps(defender_metrics_after, indent=2))
        (round_dir / "COMPLETE.json").write_text(json.dumps({"round": round_id, "status": "COMPLETE"}))

        run_summary["rounds"].append(round_summary)
        run_summary["status"] = "RUNNING"
        summary_path.write_text(json.dumps(run_summary, indent=2))
        print(f"  Round {round_id} complete: mean_drift={mean_drift:.4f} defender_gain={defender_gain}")

    run_summary["status"] = "RUN_COMPLETE"
    run_summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary_path.write_text(json.dumps(run_summary, indent=2))
    print(f"\nPhase 2 pilot complete: {attacker_name} seed={seed} → {out_dir}")
    return run_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacker", choices=ATTACKER_CHOICES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reset", action="store_true", help="reset attacker state each round (no carry)")
    parser.add_argument("--results-dir", default="results/arena")
    args = parser.parse_args()

    carry_state = not args.reset
    mode = "reset" if args.reset else "carry"
    run_id = f"phase2_{args.attacker}_{mode}_seed{args.seed}"
    out_dir = Path(args.results_dir) / run_id

    print(f"Phase 2 pilot: {args.attacker} | seed={args.seed} | carry_state={carry_state}")
    print(f"Output: {out_dir}")

    run_phase2_pilot(
        attacker_name=args.attacker,
        seed=args.seed,
        carry_state=carry_state,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
