from __future__ import annotations

import csv, json, subprocess, sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"results"/"arena"
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

@app.get("/api/health")
def health(): return {"status":"ok","mode":"local_offline"}

@app.get("/api/runs")
def runs():
    return [{"run_id":p.name,"summary":read_json(p/"run_summary.json") if (p/"run_summary.json").exists() else None}
            for p in sorted(RESULTS.glob("*")) if p.is_dir()]

@app.get("/api/runs/{run_id}")
def run(run_id: str): return read_json(run_path(run_id)/"run_summary.json")

@app.get("/api/runs/{run_id}/rounds")
def rounds(run_id: str):
    return [read_json(p/"round_summary.json") for p in sorted(run_path(run_id).glob("round_*")) if (p/"round_summary.json").exists()]

@app.get("/api/runs/{run_id}/metrics")
def metrics(run_id: str): return rounds(run_id)

@app.get("/api/runs/{run_id}/candidates")
def candidates(run_id: str):
    out=[]
    for path in sorted(run_path(run_id).glob("round_*/candidates.jsonl")):
        out.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return out

@app.get("/api/runs/{run_id}/matrix")
def matrix(run_id: str):
    path=run_path(run_id)/"final"/"arena_cross_attack_matrix.csv"
    return list(csv.DictReader(path.open()))

@app.post("/api/runs",status_code=202)
def start_run(req: RunRequest):
    if not req.confirmed: raise HTTPException(409,"resolved config must be confirmed before execution")
    if (RESULTS/req.run_id).exists(): raise HTTPException(409,"run id already exists")
    config=ROOT/"configs"/f"resolved_{req.run_id}.yaml"
    config.write_text(f"run_id: {req.run_id}\nseed: {req.seed}\nrounds: {req.rounds}\nsource_count: {req.source_count}\ntrain_count: 80\nproposal_budget: 2\nhard_example_k: 4\nattackers: [RandomSMILES, MCMC]\nadaptive_attacker: true\ndefender:\n  adapter: ridge_smoke\n")
    subprocess.Popen([sys.executable,str(ROOT/"scripts"/"run_smoke.py"),"--config",str(config)],cwd=ROOT)
    return {"run_id":req.run_id,"state":"QUEUED","resolved_config":config.read_text()}

@app.post("/api/runs/{run_id}/stop",status_code=501)
def stop(run_id: str):
    run_path(run_id)
    raise HTTPException(501,"process ownership tracking is not implemented; no process was stopped")

