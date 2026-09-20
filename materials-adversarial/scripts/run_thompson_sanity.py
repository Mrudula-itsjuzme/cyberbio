#!/usr/bin/env python3
"""Run the Thompson Sampling Search sanity benchmark with fixed regret rules."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import numpy as np

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.search import CompositeProposalOperator
from materials_adv.attacks.search.strategies import (
    ThompsonSamplingSearch, UniformSamplingSearch
)
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config

def load_predictor(path: Path, config: dict, vocab: list[str], units: str):
    import torch
    architecture = config["architecture"]
    model = TransformerRegressorModel(len(vocab), architecture["d_model"],
        architecture["n_layers"], architecture["n_heads"],
        architecture["dim_feedforward"], architecture["dropout"],
        architecture["max_seq_len"], architecture["pooling"])
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return TransformerRegressor(model, vocab, TargetScaler.load(path / "scaler.json"), units)

def get_proposals(rng, vocab, protection):
    arms = []
    sub = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=False, **protection)
    arms.append(CompositeProposalOperator([sub], rng))
    
    ins = InsertionAttack(rng, allowed_tokens=vocab, attack_budget=1, **protection)
    arms.append(CompositeProposalOperator([ins], rng))
    
    del_att = DeletionAttack(rng, attack_budget=1, **protection)
    arms.append(CompositeProposalOperator([del_att], rng))
    
    arr = RearrangementAttack(rng, window_size=3, **protection)
    arms.append(CompositeProposalOperator([arr], rng))
    return arms

def main():
    out_dir = Path("results/thompson")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "posterior_trace.csv"
    summary_path = out_dir / "thompson_sanity_summary.json"
    fixed_arms_path = out_dir / "fixed_arm_rewards.csv"
    regret_path = out_dir / "regret.csv"
    
    dataset_config, model_config, attack_config = load_config("dataset"), load_config("model"), load_config("attack")
    processed = Path(dataset_config["processed_dir"])
    dataframe = pd.read_csv(processed / "processed.csv")
    splits = json.loads((processed / "splits.json").read_text())
    vocab = json.loads((processed / "vocab.json").read_text())
    
    candidates = dataframe.iloc[splits["test"]].head(5)
    samples = [(f"row_{index}", str(row.original_representation)) for index, row in candidates.iterrows()]
    
    model_dir = Path("results/models/transformer_regressor")
    predictor = load_predictor(model_dir, model_config, vocab, dataset_config["target_units"])
    protection = attack_config["protection"]
    
    target_units = dataset_config.get("target_units", "eV")
    threshold = model_config.get("baseline_metrics", {}).get("mae")
    if threshold is None:
        raise ValueError("No verified MAE found in model config for threshold.")
        
    budgets = [10, 25]
    seeds = [42, 123, 456]
    
    all_traces = []
    results_summary = []
    fixed_arm_records = []
    regret_records = []
    
    # K documented across all experiments
    K = 50
    
    for budget in budgets:
        proposal_budget = K * budget
        
        for seed in seeds:
            print(f"Running budget={budget}, seed={seed}...")
            
            for sample_id, representation in samples:
                # 1. Hindsight Best (Independent Greedy Searches per arm)
                arm_rewards = {}
                best_arm_name = None
                best_mean_reward = -1.0
                best_drift = 0.0
                
                # Get arms fresh
                arms_for_fixed = get_proposals(np.random.default_rng(seed), vocab, protection)
                for arm in arms_for_fixed:
                    arm_name = arm.names[0]
                    rng_arm = np.random.default_rng(seed)
                    ind_search = UniformSamplingSearch(
                        predictor=predictor, proposals=[arm], rng=rng_arm,
                        query_budget=budget, proposal_budget=proposal_budget,
                        require_plausible=True, reward_mode="binary", threshold=threshold
                    )
                    res = ind_search.search(representation)
                    
                    rewards = [e.get("reward", 0.0) for e in res.trace]
                    total_reward = sum(rewards)
                    
                    # Normalize by the number of iterations it actually ran (up to proposal_budget)
                    # We compute per-step reward relative to the max_proposals budget to be fair,
                    # since arms that duplicate loop will consume proposal budget.
                    # Wait, T in Regret(T) is usually the number of arm pulls (iterations).
                    # We will use T = len(res.trace).
                    T = len(res.trace)
                    mean_reward = total_reward / T if T > 0 else 0.0
                    
                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        best_arm_name = arm_name
                    if res.best_drift > best_drift:
                        best_drift = res.best_drift
                        
                    for i, r in enumerate(rewards):
                        fixed_arm_records.append({
                            "sample_id": sample_id, "budget": budget, "seed": seed,
                            "arm": arm_name, "iteration": i+1, "reward": r,
                            "cumulative_reward": sum(rewards[:i+1])
                        })

                # 2. Uniform Selection
                rng_unif = np.random.default_rng(seed)
                arms_unif = get_proposals(rng_unif, vocab, protection)
                unif_search = UniformSamplingSearch(
                    predictor=predictor, proposals=arms_unif, rng=rng_unif,
                    query_budget=budget, proposal_budget=proposal_budget,
                    require_plausible=True, reward_mode="binary", threshold=threshold
                )
                res_unif = unif_search.search(representation)
                
                # 3. Thompson Selection
                rng_thomp = np.random.default_rng(seed)
                arms_thomp = get_proposals(rng_thomp, vocab, protection)
                thomp_search = ThompsonSamplingSearch(
                    predictor=predictor, proposals=arms_thomp, rng=rng_thomp,
                    query_budget=budget, proposal_budget=proposal_budget,
                    require_plausible=True, reward_mode="binary", threshold=threshold
                )
                res_thomp = thomp_search.search(representation)
                
                # Collect Traces
                def process_trace(trace, controller_name):
                    cum_reward = 0.0
                    for i, entry in enumerate(trace):
                        entry_copy = dict(entry)
                        entry_copy.update({
                            "sample_id": sample_id, "budget": budget, "seed": seed, "controller": controller_name
                        })
                        all_traces.append(entry_copy)
                        
                        # Empirical Pseudo-Regret
                        cum_reward += entry.get("reward", 0.0)
                        T_current = i + 1
                        opt_reward = T_current * best_mean_reward
                        regret = opt_reward - cum_reward
                        regret_records.append({
                            "sample_id": sample_id, "budget": budget, "seed": seed,
                            "controller": controller_name, "iteration": T_current,
                            "cumulative_reward": cum_reward,
                            "optimal_cumulative_reward": opt_reward,
                            "empirical_pseudo_regret": regret,
                            "target_units": target_units
                        })

                process_trace(res_unif.trace, "uniform")
                process_trace(res_thomp.trace, "thompson")
                    
                results_summary.append({
                    "sample_id": sample_id,
                    "budget": budget,
                    "seed": seed,
                    "hindsight_best_drift": best_drift,
                    "hindsight_best_arm": best_arm_name,
                    "uniform_drift": res_unif.best_drift,
                    "thompson_drift": res_thomp.best_drift,
                    "uniform_reward_sum": sum(e.get("reward", 0) for e in res_unif.trace),
                    "thompson_reward_sum": sum(e.get("reward", 0) for e in res_thomp.trace),
                    "target_units": target_units,
                    "threshold_used": threshold
                })

    df_trace = pd.DataFrame(all_traces)
    df_trace.to_csv(trace_path, index=False)
    pd.DataFrame(fixed_arm_records).to_csv(fixed_arms_path, index=False)
    pd.DataFrame(regret_records).to_csv(regret_path, index=False)
    
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"Sanity benchmark complete. Wrote traces to {out_dir}")

if __name__ == "__main__":
    main()
