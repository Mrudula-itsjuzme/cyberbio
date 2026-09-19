import argparse
import itertools
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

import sys
comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from src.core.budgets import BudgetManager
from src.attackers.evolutionary import EvolutionaryAttacker
from src.adapters.canonical_model import CanonicalModelAdapter
from src.adapters.canonical_validator import CanonicalValidatorAdapter
from src.adapters.canonical_dataset import CanonicalDatasetAdapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-budget", type=int, default=50)
    args = parser.parse_args()

    dataset = CanonicalDatasetAdapter()
    frozen_set = set(dataset.source_ids)
    all_indices = set(range(len(dataset.df)))
    train_ids = list(all_indices - frozen_set)
    
    rng = np.random.default_rng(999)
    val_ids = rng.choice(train_ids, size=10, replace=False)
    sources = [dataset.df.iloc[i][dataset.seq_col] for i in val_ids]

    model = CanonicalModelAdapter()
    validator = CanonicalValidatorAdapter()
    
    pop_sizes = [5, 10, 20]
    mut_rates = [0.2, 0.5, 0.8]
    elitism = [1, 2]
    
    out_dir = Path(__file__).resolve().parent.parent / "results/evolutionary_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_config = None
    best_score = -1.0
    
    all_results = []
    
    for p, m, e in itertools.product(pop_sizes, mut_rates, elitism):
        print(f"Testing Config - pop: {p}, mut: {m}, elitism: {e}")
        drifts = []
        for i, (sid, seq) in enumerate(zip(val_ids, sources)):
            budget = BudgetManager(max_queries=args.query_budget)
            attacker = EvolutionaryAttacker(population_size=p, mutation_rate=m, elitism_count=e)
            res = attacker.attack(sid, seq, model, validator, budget, rng)
            drifts.append(res.prediction_drift)
            
        mean_drift = float(np.mean(drifts))
        print(f" -> Mean Drift: {mean_drift:.4f}")
        
        all_results.append({
            "population_size": p,
            "mutation_rate": m,
            "elitism_count": e,
            "mean_drift": mean_drift
        })
        
        if mean_drift > best_score:
            best_score = mean_drift
            best_config = {"population_size": p, "mutation_rate": m, "elitism_count": e}
            
            # Save the generation history of the best config
            best_history = res.metadata.get("generation_history", [])
            
    with open(out_dir / "evolutionary_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
        
    print(f"Selected Config: {best_config}")
    
    # Generate convergence plot for the best run (last source of the best config as representative)
    if best_history:
        gens = list(range(len(best_history)))
        best_drifts = [h["best_drift"] for h in best_history]
        mean_fits = [h["mean_fitness"] for h in best_history]
        
        plt.figure()
        plt.plot(gens, best_drifts, label="Best Drift")
        plt.plot(gens, mean_fits, label="Mean Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Drift")
        plt.title(f"Evolutionary Convergence (Best Config)")
        plt.legend()
        plt.savefig(out_dir / "convergence.png")
        plt.close()

if __name__ == "__main__":
    main()
