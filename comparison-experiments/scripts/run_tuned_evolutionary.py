import subprocess
import argparse
import itertools
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-count", type=int, default=100)
    args = parser.parse_args()

    attackers = ["evolutionary"]
    budgets = [10, 20, 50]
    seeds = [42, 123, 2026, 777, 999]
    
    script = Path(__file__).resolve().parent / "run_comparison.py"
    base_out = Path(__file__).resolve().parent.parent / "results/raw/tuned_evolutionary_sweep"
    
    for attacker, budget, seed in itertools.product(attackers, budgets, seeds):
        out_dir = base_out / attacker / f"budget_{budget}" / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python", str(script),
            "--attack", attacker,
            "--source-count", str(args.source_count),
            "--query-budget", str(budget),
            "--seed", str(seed),
            "--output-dir", str(out_dir)
        ]
        
        print(f"Running Tuned: {attacker} | budget: {budget} | seed: {seed}")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
