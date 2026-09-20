from __future__ import annotations

import csv, json, subprocess, sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"results"/"arena"
FINAL=ROOT/"results"/"final"
app=FastAPI(title="CyberBio Adversarial Arena",docs_url="/api/docs")

class RunRequest(BaseModel):
    run_id: str=Field(pattern=r"^[A-Za-z0-9_-]+$")
    rounds: int=Field(default=2,ge=1,le=5)
    source_count: int=Field(default=5,ge=1,le=20)
    seed: int=20260920
    confirmed: bool=False

def run_path(run_id: str) -> Path:
    if not run_id.replace("-","").replace("_","").isalnum(): raise HTTPException(400,"invalid run id")
    path=RESULTS/run_id
    if not path.is_dir(): raise HTTPException(404,"run not found")
    return path

def read_json(path: Path):
    if not path.exists(): raise HTTPException(404,f"artifact not found: {path.name}")
    return json.loads(path.read_text())

def read_csv(path: Path):
    if not path.exists(): return []
    return list(csv.DictReader(path.open()))

@app.get("/api/health")
def health(): return {"status":"ok","mode":"local_offline"}

@app.get("/api/runs")
def runs():
    runs = [{"run_id":p.name,"summary":read_json(p/"run_summary.json") if (p/"run_summary.json").exists() else None}
            for p in sorted(RESULTS.glob("*")) if p.is_dir()]
    # Return AGGREGATE as a special run ID
    if (FINAL/"arena_multiseed_rounds.csv").exists():
        runs.insert(0, {"run_id": "AGGREGATE", "summary": {"status": "AGGREGATE", "scientific_status": "MULTI-SEED AGGREGATE"}})
    return runs

@app.get("/api/runs/{run_id}")
def run(run_id: str):
    if run_id == "AGGREGATE":
        return {"status": "AGGREGATE", "scientific_status": "MULTI-SEED AGGREGATE", "rounds": []}
    return read_json(run_path(run_id)/"run_summary.json")

@app.get("/api/runs/{run_id}/data")
def run_data(run_id: str):
    if run_id == "AGGREGATE":
        return {
            "rounds": read_csv(FINAL/"arena_multiseed_rounds.csv"),
            "transfer": read_csv(FINAL/"arena_multiseed_transfer.csv"),
            "selection_gain": read_csv(FINAL/"arena_multiseed_selection_gain.csv"),
            "thompson": read_csv(FINAL/"arena_multiseed_thompson.csv"),
            "redistribution": read_csv(FINAL/"arena_robustness_redistribution.csv"),
            "matrix": read_csv(FINAL/"arena_multiseed_rounds.csv"), # Placeholder for aggregate matrix logic in frontend
        }
    
    # Single seed
    rp = run_path(run_id)
    return {
        "rounds": read_csv(rp/"final"/"arena_round_summary.csv"),
        "transfer": read_csv(FINAL/"arena_pilot_transfer.csv") if (FINAL/"arena_pilot_transfer.csv").exists() else [],
        "selection_gain": read_csv(FINAL/"arena_pilot_selection_gain.csv") if (FINAL/"arena_pilot_selection_gain.csv").exists() else [],
        "thompson": read_csv(FINAL/"arena_pilot_thompson.csv") if (FINAL/"arena_pilot_thompson.csv").exists() else [],
        "matrix": read_csv(rp/"final"/"arena_cross_attack_matrix.csv"),
    }

@app.get("/api/aggregate/redistribution")
def redistribution():
    return read_csv(FINAL/"arena_robustness_redistribution.csv")

@app.get("/api/aggregate/tradeoff")
def tradeoff():
    return read_csv(FINAL/"arena_clean_robust_tradeoff.csv")

@app.get("/api/aggregate/summary")
def aggregate_summary():
    return {
        "rounds": read_csv(FINAL/"arena_multiseed_rounds.csv"),
        "transfer": read_csv(FINAL/"arena_multiseed_transfer.csv"),
        "selection_gain": read_csv(FINAL/"arena_multiseed_selection_gain.csv"),
        "thompson": read_csv(FINAL/"arena_multiseed_thompson.csv"),
        "redistribution": read_csv(FINAL/"arena_robustness_redistribution.csv"),
        "tradeoff": read_csv(FINAL/"arena_clean_robust_tradeoff.csv"),
    }

@app.post("/api/runs/{run_id}/stop",status_code=501)
def stop(run_id: str):
    run_path(run_id)
    raise HTTPException(501,"process ownership tracking is not implemented; no process was stopped")

@app.get("/api/runs/{run_id}/candidates")
def candidates(run_id: str):
    if run_id == "AGGREGATE":
        # Load from all seed runs? Just return [] for simplicity
        return []
    out=[]
    rp = run_path(run_id)
    hard = set()
    for path in sorted(rp.glob("round_*/hard_examples.jsonl")):
        hard.update([json.loads(line).get("source_id") for line in path.read_text().splitlines() if line])
        
    for round_id, path in enumerate(sorted(rp.glob("round_*/candidates.jsonl"))):
        for line in path.read_text().splitlines():
            if line:
                c = json.loads(line)
                c["round"] = round_id
                c["is_hard_example"] = c.get("source_id") in hard
                c["held_out_eval"] = (c.get("attack_family") == "Evolutionary")
                c["seed"] = rp.name.replace("arena_transformer_seed", "")
                c["defender_checkpoint"] = f"D_{round_id}"
                out.append(c)
    return out

@app.get("/api/runs/{run_id}/matrix")
def matrix(run_id: str):
    if run_id == "AGGREGATE": return []
    path=run_path(run_id)/"final"/"arena_cross_attack_matrix.csv"
    return list(csv.DictReader(path.open()))

