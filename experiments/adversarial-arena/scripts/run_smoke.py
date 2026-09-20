from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/"materials-adversarial"/"src")]

from arena.adaptive import ThompsonController
from arena.attackers import mcmc_adapter, random_smiles_adapter
from arena.contracts import SourceSample
from arena.defenders import RidgeSmokeDefender
from arena.orchestration import ArenaRunner
from arena.storage import RunStorage
from arena.validators import MaterialsValidator

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default=str(HERE/"configs"/"smoke.yaml")); parser.add_argument("--run-id")
    args=parser.parse_args(); config=yaml.safe_load(Path(args.config).read_text())
    if args.run_id: config["run_id"]=args.run_id
    df=pd.read_csv(ROOT/"materials-adversarial"/"data"/"processed"/"processed.csv")
    def sample(rows): return [SourceSample(str(r.polymer_id),r.original_representation,float(r.property_value)) for r in rows.itertuples()]
    train=sample(df[df.split=="train"].head(config["train_count"])); validation=sample(df[df.split=="val"].head(config["source_count"]))
    vocab=json.loads((ROOT/"materials-adversarial"/"data"/"processed"/"vocab.json").read_text())
    seed=config["seed"]; attackers={"RandomSMILES":random_smiles_adapter(seed,config["proposal_budget"]),
        "MCMC":mcmc_adapter(seed+1,vocab,steps=5,proposal_budget=1)}
    storage=RunStorage(HERE/"results"/"arena",config["run_id"])
    runner=ArenaRunner(defender=RidgeSmokeDefender(seed),attackers=attackers,validator=MaterialsValidator(),
        storage=storage,seed=seed,hard_example_k=config["hard_example_k"],controller=ThompsonController(list(attackers),seed))
    print(json.dumps(runner.run(clean_train=train,clean_validation=validation,rounds=config["rounds"],config=config),indent=2))

if __name__=="__main__": main()

