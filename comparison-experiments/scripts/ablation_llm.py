import subprocess
from pathlib import Path
import os

def main():
    script = Path(__file__).resolve().parent / "run_comparison.py"
    base_out = Path(__file__).resolve().parent.parent / "results/raw/llm_ablation"
    
    modes = ["blind", "objective-aware", "iterative", "operator-constrained"]
    budget = 10
    seed = 42
    source_count = 5
    
    for mode in modes:
        out_dir = base_out / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python", str(script),
            "--attack", f"llm_{mode}",
            "--source-count", str(source_count),
            "--query-budget", str(budget),
            "--seed", str(seed),
            "--output-dir", str(out_dir),
            "--use-real-llm"
        ]
        
        print(f"Running LLM Ablation Pilot: {mode}")
        subprocess.run(cmd)
        
if __name__ == "__main__":
    main()
