import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Adjust path to import comparison framework modules
comp_exp_root = Path(__file__).resolve().parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

from src.core.budgets import BudgetManager
from src.attackers.random_search import RandomSearchAttacker
from src.attackers.evolutionary import EvolutionaryAttacker
from src.attackers.llm_guided import LLMGuidedAttacker, MockLLMProvider
from src.adapters.canonical_model import CanonicalModelAdapter
from src.adapters.canonical_validator import CanonicalValidatorAdapter
from src.adapters.canonical_dataset import CanonicalDatasetAdapter
from src.adapters.canonical_mcmc import CanonicalMCMCAdapter
from src.metrics.statistics import compute_comparison_statistics

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", nargs="+", choices=["random", "mcmc", "evolutionary", "llm", "llm_blind", "llm_objective-aware", "llm_iterative", "llm_operator-constrained", "all"], default=["random"])
    parser.add_argument("--use-real-llm", action="store_true", help="Use real OpenAICompatibleProvider instead of Mock")
    parser.add_argument("--source-count", type=int, default=20)
    parser.add_argument("--query-budget", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args(args)

def main():
    args = parse_args()
    
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    import sys
    sys.path.append(str(Path(__file__).parent))
    from generate_manifest import generate_manifest
    repo_root = Path(__file__).resolve().parent.parent.parent / "materials-adversarial"
    model_chk = repo_root / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
    source_ids = Path(__file__).resolve().parent.parent / "data" / "frozen_source_ids.json"
    generate_manifest(out_dir, args, model_chk, source_ids)

    print("Loading Canonical Environment...")
    model = CanonicalModelAdapter()
    validator = CanonicalValidatorAdapter()
    dataset = CanonicalDatasetAdapter()
    
    sources = dataset.get_frozen_sources()[:args.source_count]
    
    attackers = []
    attacks_to_run = ["random", "mcmc", "evolutionary", "llm"] if "all" in args.attack else args.attack
    if "random" in attacks_to_run:
        attackers.append(RandomSearchAttacker())
    if "mcmc" in attacks_to_run:
        attackers.append(CanonicalMCMCAdapter())
    if "evolutionary" in attacks_to_run:
        config_path = comp_exp_root / "results/evolutionary_tuning/evolutionary_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                evo_cfg = json.load(f)
            attackers.append(EvolutionaryAttacker(**evo_cfg))
        else:
            attackers.append(EvolutionaryAttacker())
            
    from src.attackers.llm_providers import OpenAICompatibleProvider, MockLLMProvider
    
    provider = OpenAICompatibleProvider() if getattr(args, "use_real_llm", False) else MockLLMProvider(mode="valid")
    
    # LLM routing
    for a in attacks_to_run:
        if a.startswith("llm_"):
            mode = a.split("llm_")[1]
            attackers.append(LLMGuidedAttacker(provider=provider, prompt_mode=mode))
        elif a == "llm":
            attackers.append(LLMGuidedAttacker(provider=provider, prompt_mode="blind"))
        
    all_stats = []
    
    for attacker in attackers:
        print(f"\n--- Running {attacker.name} ---")
        results = []
        for i, (sid, seq) in enumerate(sources):
            budget = BudgetManager(max_queries=args.query_budget, max_generations=args.query_budget * 5)
            res = attacker.attack(
                source_id=sid,
                source_sequence=seq,
                model=model,
                validator=validator,
                budget=budget,
                rng=rng
            )
            res.seed = args.seed
            results.append(res)
            print(f"  Source {i+1}/{len(sources)} | queries: {res.query_count} | drift: {res.prediction_drift:.4f}")
            
        # Save raw
        raw_path = out_dir / f"raw_{attacker.name}_s{args.seed}.jsonl"
        with open(raw_path, "w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")
                
        # Stats
        stats = compute_comparison_statistics(results)
        all_stats.append(stats)
        
    print("\n" + "="*80)
    print(f"{'Attacker':<20} | {'Mean Drift':<10} | {'Med Drift':<9} | {'P90 Drift':<9} | {'Succ Rate':<9} | {'Valid Rate':<10} | {'Cnst Pass':<9} | {'Queries':<7} | {'Runtime':<7}")
    print("-" * 115)
    
    # Generate canonical aggregated results file
    import pandas as pd
    all_results = []
    for fpath in out_dir.glob("raw_*.jsonl"):
        with open(fpath, "r") as f:
            for line in f:
                all_results.append(json.loads(line))
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / "aggregated_results.csv", index=False)
        print(f"\nCanonical aggregated results saved to: {out_dir / 'aggregated_results.csv'}")
    for s in all_stats:
        print(f"{s['attacker']:<20} | {s['mean_drift']:<10.4f} | {s['median_drift']:<9.4f} | {s['p90_drift']:<9.4f} | {s['success_rate']:<9.4f} | {s['valid_rdkit_rate']:<10.4f} | {s['constraint_pass_rate']:<9.4f} | {s['mean_queries']:<7.1f} | {s['mean_runtime']:<7.2f}")
        
if __name__ == "__main__":
    main()

