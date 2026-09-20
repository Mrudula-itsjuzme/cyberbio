from __future__ import annotations

import argparse, json, sys, shutil
from pathlib import Path
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/"materials-adversarial"/"src")]

from arena.adaptive import ThompsonController
from arena.attackers import mcmc_adapter, random_smiles_adapter, evolutionary_adapter
from arena.contracts import SourceSample
from arena.transpolymer import TransPolymerDefender
from arena.orchestration import ArenaRunner
from arena.storage import RunStorage
from arena.validators import MaterialsValidator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE/"configs"/"transformer_pilot.yaml"))
    parser.add_argument("--run-id", default="pilot_run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--curriculum")
    parser.add_argument("--resume", action="store_true", help="Resume from stopped run")
    args = parser.parse_args()
    
    config = yaml.safe_load(Path(args.config).read_text())
    run_id = args.run_id
    if args.seed: config["seeds"] = [args.seed]
    if args.curriculum: config["primary_curriculum"] = args.curriculum
    
    df = pd.read_csv(ROOT/"materials-adversarial"/"data"/"processed"/"processed.csv")
    def sample(rows): return [SourceSample(str(r.polymer_id),r.original_representation,float(r.property_value)) for r in rows.itertuples()]
    
    # Pilot specifies 10-20 fixed test sources, but we need some train sources too.
    # We will use 100 for training, and source_count for validation.
    train = sample(df[df.split=="train"].head(100))
    validation = sample(df[df.split=="val"].head(config["source_count"]))
    
    vocab = json.loads((ROOT/"materials-adversarial"/"data"/"processed"/"vocab.json").read_text())
    
    # Start loop for seeds and curricula if needed, or just run the primary one.
    seed = config["seeds"][0]
    curriculum = config["primary_curriculum"]
    
    q_budget = config["query_budget"]
    attackers = {}
    if curriculum in ["fixed_randomsmiles", "thompson", "thompson_leave_one_out"]:
        attackers["RandomSMILES"] = random_smiles_adapter(seed, q_budget)
    if curriculum in ["fixed_mcmc", "thompson", "thompson_leave_one_out"]:
        attackers["MCMC"] = mcmc_adapter(seed+1, vocab, steps=5, proposal_budget=q_budget)
    if curriculum in ["fixed_evolutionary", "thompson", "thompson_leave_one_out"]:
        attackers["Evolutionary"] = evolutionary_adapter(seed+2, query_budget=q_budget)
        
    storage = RunStorage(HERE/"results"/"arena", run_id)
    
    # We need paths to canonical model
    checkpoint_dir = ROOT / "materials-adversarial" / "results" / "models" / "transformer_regressor"
    scaler_path = ROOT / "materials-adversarial" / "data" / "processed" / "scaler.pkl"
    config_path = ROOT / "materials-adversarial" / "configs" / "transformer.yaml"
    
    defender = TransPolymerDefender(

        ROOT/"materials-adversarial",
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        lambda_rep=config["defender"]["lambda_representation"],
        learning_rate=config["defender"]["learning_rate"],
        epochs=config["defender"]["epochs_per_round"]
    )
    
    controller = ThompsonController([a for a in attackers if a != "Evolutionary"], seed) if curriculum == "thompson_leave_one_out" else (ThompsonController(list(attackers), seed) if curriculum == "thompson" else None)
    
    config["real_pilot"] = True
    
    runner = ArenaRunner(
        defender=defender,
        attackers=attackers,
        validator=MaterialsValidator(),
        storage=storage,
        seed=seed,
        hard_example_k=config["hard_example_k"],
        controller=controller
    )
    
    print(json.dumps(runner.run(clean_train=train, clean_validation=validation, rounds=config["rounds"], config=config, resume=args.resume), indent=2))

if __name__=="__main__": main()
