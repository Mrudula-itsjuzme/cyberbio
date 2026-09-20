"""
Compute True Attacker Adaptation Gain G_adapt(t).

For each Phase 2 run and each round t:
  G_adapt(t) = R(A_{t+1}, D_t) - R(A_t, D_t)
  
R(A_t, D_t) is already computed during the run (it's the round's mean_drift).
This script loads D_t and A_{t+1}, runs the attacker (without further carrying state),
and computes R(A_{t+1}, D_t).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SRC = HERE.parent.parent / "materials-adversarial" / "src"
sys.path[:0] = [str(HERE), str(SRC)]

from arena.probabilistic_attackers import cem_attacker, smc_attacker, validity_smc_attacker
from arena.transpolymer import TransPolymerDefender
from scripts.run_phase2_pilot import _load_sources, QUERY_BUDGET, N_SOURCES

def load_attacker_state(attacker_name: str, seed: int, state_file: Path):
    factories = {
        "cem": cem_attacker,
        "smc": smc_attacker,
        "validity_smc": validity_smc_attacker,
    }
    # Create with carry_state=True so it accepts the loaded state,
    # but we will only run it once and discard the updated state.
    attacker = factories[attacker_name](seed=seed, query_budget=QUERY_BUDGET, carry_state=True)
    
    saved = json.loads(state_file.read_text())
    attacker._round_id = saved.get("round_id", 0)
    if attacker_name == "cem" and "probs" in saved:
        attacker._probs = saved["probs"]
    elif attacker_name in ("smc", "validity_smc"):
        if "particle_state" in saved:
            attacker._particle_state = saved.get("particle_state")
            attacker._weight_state = saved.get("weight_state")
    
    return attacker

def main():
    results_dir = HERE / "results" / "arena"
    runs = list(results_dir.glob("phase2_*_carry_seed*"))
    
    checkpoint_dir = SRC.parent / "results" / "models" / "transformer_regressor"
    
    # Store G_adapt results
    g_adapt_results = []
    
    for run_dir in runs:
        parts = run_dir.name.split("_")
        if "cem" in parts:
            attacker_name = "cem"
        elif "validity" in parts:
            attacker_name = "validity_smc"
        else:
            attacker_name = "smc"
            
        seed_str = parts[-1].replace("seed", "")
        seed = int(seed_str)
        
        print(f"\nProcessing {run_dir.name}...")
        
        # Load sources
        sources = _load_sources(seed, N_SOURCES)
        
        for t in range(3):
            round_dir = run_dir / f"round_{t:03d}"
            if not round_dir.exists():
                continue
                
            summary = json.loads((round_dir / "round_summary.json").read_text())
            R_At_Dt = summary["mean_drift"]
            
            # Check if we already computed it
            if "g_adapt" in summary:
                g_adapt_results.append({
                    "run": run_dir.name,
                    "attacker": attacker_name,
                    "seed": seed,
                    "round": t,
                    "R_At_Dt": R_At_Dt,
                    "R_At1_Dt": summary.get("R_At1_Dt"),
                    "G_adapt": summary["g_adapt"]
                })
                print(f"  Round {t}: G_adapt already computed ({summary['g_adapt']:.4f})")
                continue
                
            # Load D_t
            defender = TransPolymerDefender(
                project_root=SRC.parent,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
                lambda_rep=0.5,
                learning_rate=0.001,
                epochs=1
            )
            if t > 0:
                ckpt = run_dir / f"round_{t - 1:03d}" / "defender_checkpoint.json"
                defender.load_checkpoint(str(ckpt))
                
            # Load A_{t+1}
            state_file = round_dir / f"{attacker_name}_state.json"
            if not state_file.exists():
                print(f"  Round {t}: {state_file.name} missing, skipping.")
                continue
                
            attacker = load_attacker_state(attacker_name, seed, state_file)
            
            # Run A_{t+1} against D_t
            print(f"  Round {t}: Evaluating A_{t+1} against D_t...")
            records = attacker.propose(sources, defender)
            drifts = [r.drift for r in records if r.drift is not None]
            R_At1_Dt = sum(drifts) / len(drifts) if drifts else 0.0
            
            G_adapt = R_At1_Dt - R_At_Dt
            print(f"    R(A_t, D_t)   = {R_At_Dt:.4f}")
            print(f"    R(A_t+1, D_t) = {R_At1_Dt:.4f}")
            print(f"    G_adapt(t)    = {G_adapt:.4f}")
            
            # Update summary
            summary["R_At1_Dt"] = R_At1_Dt
            summary["g_adapt"] = G_adapt
            (round_dir / "round_summary.json").write_text(json.dumps(summary, indent=2))
            
            g_adapt_results.append({
                "run": run_dir.name,
                "attacker": attacker_name,
                "seed": seed,
                "round": t,
                "R_At_Dt": R_At_Dt,
                "R_At1_Dt": R_At1_Dt,
                "G_adapt": G_adapt
            })
            
    # Save CSV
    import pandas as pd
    if g_adapt_results:
        df = pd.DataFrame(g_adapt_results)
        df.to_csv(HERE / "results" / "arena" / "arena_phase2_g_adapt_raw.csv", index=False)
        print("\nSaved raw G_adapt results to arena_phase2_g_adapt_raw.csv")

if __name__ == "__main__":
    main()
