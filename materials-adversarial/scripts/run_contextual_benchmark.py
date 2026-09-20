#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from pathlib import Path

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.search import CompositeProposalOperator
from materials_adv.attacks.search.strategies import (
    ThompsonSamplingSearch, UniformSamplingSearch, ContextualThompsonSamplingSearch
)
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config
from materials_adv.attacks.search.context import ContextFeatureExtractor, ContextScaler

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

def train_contextual_prior(train_data: pd.DataFrame, feature_names: list[str]):
    # Train scaler
    features = train_data[[f"feat_{f}" for f in feature_names]].values
    scaler = ContextScaler()
    scaler.fit(features)
    scaled_features = scaler.transform(features)
    
    mu_priors = {}
    cov_priors = {}
    
    # Simple Bayesian Ridge Regression with numpy
    # alpha is precision of the noise, lambda_ is precision of the weights
    alpha = 1.0
    lambda_ = 1.0
    
    for arm in train_data["arm"].unique():
        arm_df = train_data[train_data["arm"] == arm]
        idx = arm_df.index
        X = scaled_features[idx]
        y = arm_df["mean_reward"].values
        
        # prec = lambda * I + alpha * X^T X
        prec = lambda_ * np.eye(X.shape[1]) + alpha * (X.T @ X)
        cov = np.linalg.inv(prec)
        
        # mu = cov @ (alpha * X^T y)
        mu = cov @ (alpha * X.T @ y)
        
        mu_priors[arm] = mu
        cov_priors[arm] = cov
        
    return scaler, mu_priors, cov_priors

def main():
    out_dir = Path("results/contextual_thompson")
    split_path = out_dir / "context_source_split.json"
    train_data_path = out_dir / "context_arm_training_data.csv"
    test_results_path = out_dir / "test_results.csv"
    
    if not train_data_path.exists():
        print("Training data not found.")
        return
        
    with open(split_path) as f:
        splits = json.load(f)
    test_ids = set(splits["test_ids"])
    
    dataset_config = load_config("dataset")
    processed = Path(dataset_config["processed_dir"])
    df = pd.read_csv(processed / "processed.csv")
    vocab = json.loads((processed / "vocab.json").read_text())
    
    model_config = load_config("model")
    model_dir = Path("results/models/transformer_regressor")
    predictor = load_predictor(model_dir, model_config, vocab, dataset_config["target_units"])
    
    attack_config = load_config("attack")
    protection = attack_config["protection"]
    threshold = model_config.get("baseline_metrics", {}).get("mae", 0.4619)
    
    extractor = ContextFeatureExtractor(vocab)
    train_data = pd.read_csv(train_data_path)
    
    # 1. Train the context prior
    print("Training context prior...")
    scaler, mu_priors, cov_priors = train_contextual_prior(train_data, extractor.feature_names)
    
    # Global best fixed arm
    global_arm_rewards = train_data.groupby("arm")["mean_reward"].mean()
    global_best_arm = global_arm_rewards.idxmax()
    print(f"Global best arm: {global_best_arm}")
    
    test_df = df[df.index.map(lambda i: f"row_{i}" in test_ids)]
    print(f"Running benchmark on {len(test_df)} test sources...")
    
    budgets = [5, 10, 25]
    seed = splits["seed"]
    K = 50
    
    all_results = []
    
    for idx, row in test_df.iterrows():
        sample_id = f"row_{idx}"
        rep = str(row.original_representation)
        features = extractor.extract(rep)
        scaled_features = scaler.transform(features)
        
        # Shuffled context for ablation
        rng_shuffle = np.random.default_rng(seed + idx + 999)
        shuffled_features = rng_shuffle.permutation(scaled_features)
        
        for budget in budgets:
            proposal_budget = K * budget
            
            # Hindsight Oracle
            rng_oracle = np.random.default_rng(seed + idx)
            arms_for_oracle = get_proposals(rng_oracle, vocab, protection)
            oracle_rewards = {}
            for arm in arms_for_oracle:
                name = arm.names[0]
                res = UniformSamplingSearch(
                    predictor=predictor, proposals=[arm], rng=np.random.default_rng(seed + idx),
                    query_budget=budget, proposal_budget=proposal_budget,
                    require_plausible=True, reward_mode="binary", threshold=threshold
                ).search(rep)
                oracle_rewards[name] = sum(e.get("reward", 0.0) for e in res.trace)
            hindsight_best_arm = max(oracle_rewards, key=oracle_rewards.get)
            hindsight_best_reward = oracle_rewards[hindsight_best_arm]
            
            # Run Controllers
            def run_controller(c_type, feats=None):
                rng_ctrl = np.random.default_rng(seed + idx)
                arms = get_proposals(rng_ctrl, vocab, protection)
                if c_type == "uniform":
                    search = UniformSamplingSearch(predictor, arms, rng_ctrl, query_budget=budget, proposal_budget=proposal_budget, require_plausible=True, reward_mode="binary", threshold=threshold)
                elif c_type == "thompson":
                    search = ThompsonSamplingSearch(predictor, arms, rng_ctrl, query_budget=budget, proposal_budget=proposal_budget, require_plausible=True, reward_mode="binary", threshold=threshold)
                elif c_type == "contextual":
                    search = ContextualThompsonSamplingSearch(predictor, arms, rng_ctrl, query_budget=budget, proposal_budget=proposal_budget, context_features=feats, mu_priors=mu_priors, cov_priors=cov_priors, require_plausible=True, reward_mode="binary", threshold=threshold)
                elif c_type == "global_best":
                    best_arms = [a for a in arms if a.names[0] == global_best_arm]
                    search = UniformSamplingSearch(predictor, best_arms, rng_ctrl, query_budget=budget, proposal_budget=proposal_budget, require_plausible=True, reward_mode="binary", threshold=threshold)
                
                res = search.search(rep)
                return sum(e.get("reward", 0.0) for e in res.trace)
                
            reward_unif = run_controller("uniform")
            reward_thomp = run_controller("thompson")
            reward_ctx = run_controller("contextual", scaled_features)
            reward_shuf = run_controller("contextual", shuffled_features)
            reward_global = run_controller("global_best")
            
            all_results.append({
                "sample_id": sample_id,
                "budget": budget,
                "hindsight_best_arm": hindsight_best_arm,
                "hindsight_best_reward": hindsight_best_reward,
                "uniform_reward": reward_unif,
                "thompson_reward": reward_thomp,
                "contextual_reward": reward_ctx,
                "shuffled_context_reward": reward_shuf,
                "global_best_reward": reward_global
            })
            
    pd.DataFrame(all_results).to_csv(test_results_path, index=False)
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
