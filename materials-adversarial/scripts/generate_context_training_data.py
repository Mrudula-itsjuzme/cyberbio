#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from pathlib import Path

from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.search import CompositeProposalOperator
from materials_adv.attacks.search.strategies import UniformSamplingSearch
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.data.scaler import TargetScaler
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.utils.config import load_config
from materials_adv.attacks.search.context import ContextFeatureExtractor

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
    out_dir = Path("results/contextual_thompson")
    split_path = out_dir / "context_source_split.json"
    train_data_path = out_dir / "context_arm_training_data.csv"
    
    with open(split_path) as f:
        splits = json.load(f)
    train_ids = set(splits["train_ids"])
    
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
    
    # We will use budget=10 to gather training data as we care about early-budget preferences.
    budget = 10
    K = 50
    proposal_budget = K * budget
    seed = splits["seed"]
    
    records = []
    
    # Filter df to only train_ids
    train_df = df[df.index.map(lambda i: f"row_{i}" in train_ids)]
    print(f"Generating training data for {len(train_df)} train sources...")
    
    for idx, row in train_df.iterrows():
        sample_id = f"row_{idx}"
        rep = str(row.original_representation)
        features = extractor.extract(rep)
        
        # We need a fresh RNG for each source so that arms are evaluated fairly
        rng_source = np.random.default_rng(seed + idx)
        arms_for_fixed = get_proposals(rng_source, vocab, protection)
        
        for arm in arms_for_fixed:
            arm_name = arm.names[0]
            rng_arm = np.random.default_rng(seed + idx)
            ind_search = UniformSamplingSearch(
                predictor=predictor, proposals=[arm], rng=rng_arm,
                query_budget=budget, proposal_budget=proposal_budget,
                require_plausible=True, reward_mode="binary", threshold=threshold
            )
            res = ind_search.search(rep)
            
            rewards = [e.get("reward", 0.0) for e in res.trace]
            cum_reward = sum(rewards)
            T = len(res.trace)
            mean_reward = cum_reward / T if T > 0 else 0.0
            
            # Count target queries (non-duplicate, queried)
            target_queries = sum(1 for e in res.trace if e.get("queried", False) and not e.get("duplicate", False))
            
            valid_proposals = sum(1 for e in res.trace if e.get("representation_valid", False) and e.get("plausible", False))
            validity_rate = valid_proposals / T if T > 0 else 0.0
            
            record = {
                "source_id": sample_id,
                "arm": arm_name,
                "seed": seed,
                "query_budget": budget,
                "proposal_budget": proposal_budget,
                "cumulative_reward": cum_reward,
                "mean_reward": mean_reward,
                "best_drift": res.best_drift,
                "validity_rate": validity_rate,
                "target_queries": target_queries,
                "proposal_count": T
            }
            # Append features dynamically
            for i, fname in enumerate(extractor.feature_names):
                record[f"feat_{fname}"] = features[i]
                
            records.append(record)
            
    out_df = pd.DataFrame(records)
    out_df.to_csv(train_data_path, index=False)
    print(f"Saved training data to {train_data_path}")

if __name__ == "__main__":
    main()
