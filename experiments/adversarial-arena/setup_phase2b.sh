#!/bin/bash
set -e

# 1. EARLY-STOP AUDIT
cat << 'DOC' > docs/CARRY_RESET_SMC_PHASE2_EARLY_STOP_AUDIT.md
# SMC Early Stop Audit (Phase 2)

## Finding
SMC seeds 20260921 and 20260922 in Phase 2 exited after Round 1 instead of completing all 3 rounds (0, 1, 2). Despite this, they were marked as `RUN_COMPLETE` in `run_summary.json`.

## Root Cause Analysis
**ROOT_CAUSE = UNKNOWN**

Confirmed facts:
- The Phase 2 configured loop expected 3 rounds.
- No intended scientific early-stop path exists in `run_phase2_pilot.py` for SMC.
- Two SMC runs persist only R0/R1.
- The `run_summary.json` summaries later contain `RUN_COMPLETE`.
- No definitive repository evidence identifies why the loop terminated early.

## Phase 2B Resolution
To definitively prevent ambiguity in Phase 2B, `run_carry_reset.py` will implement robust completion semantics:
- `RUN_COMPLETE_FULL`
- `RUN_COMPLETE_EARLY_STOP`
- `RUN_PARTIAL`
- `RUN_FAILED`
DOC

# 2. CONFIG LOCK & RNG POLICY
cat << 'DOC' > docs/CARRY_RESET_CONFIG_LOCK.md
# Phase 2B: Carry vs Reset Config Lock

## RNG_POLICY
Reset condition:
Restore algorithmic adaptive state to initial prior before each round, but derive per-round RNG deterministically from `base_seed + round_id + source_id`.

Carry condition:
Use the exact same per-round RNG schedule as the reset condition.
This avoids giving carry/reset different randomness merely because their internal RNG consumed different numbers previously.

## REFERENCE DEFENDER DESIGN
TREATMENT-INDEPENDENT REFERENCE DEFENDER TRAJECTORY (`D_ref,t`):
Generated entirely independently of carry/reset assignment.
- source cohort locked
- hard-example rule locked
- defender initialization locked
- training settings locked
- checkpoint sequence locked
Checkpoints persisted to `reference_defender/D_ref_000`, `reference_defender/D_ref_001`, etc.
Both carry/reset attackers must be evaluated against exactly the same D_ref,t checkpoint.
Neither carry nor reset may modify D_ref,t.
DOC

# 3. STATE MANAGEMENT API
cat << 'PY' > arena/state_management.py
import json
import hashlib

def snapshot_attacker_state(attacker):
    state = {"round_id": getattr(attacker, "_round_id", 0)}
    if attacker.name == "CEM":
        state["probs"] = list(attacker._probs) if getattr(attacker, "_probs", None) is not None else None
    elif attacker.name == "SMC":
        state["particles"] = list(attacker._particle_state) if getattr(attacker, "_particle_state", None) is not None else None
        state["weights"] = list(attacker._weight_state) if getattr(attacker, "_weight_state", None) is not None else None
    elif attacker.name == "ValiditySMC":
        state["particles"] = list(attacker._particle_state) if getattr(attacker, "_particle_state", None) is not None else None
        state["weights"] = list(attacker._weight_state) if getattr(attacker, "_weight_state", None) is not None else None
        post = {}
        if hasattr(attacker, "_validity_model") and hasattr(attacker._validity_model, "counts"):
            for k, v in attacker._validity_model.counts.items():
                post[k] = {"alpha": float(v[0]), "beta": float(v[1])}
        state["validity_posterior"] = post
    return state

def restore_attacker_state(attacker, snap):
    attacker._round_id = snap["round_id"]
    if attacker.name == "CEM":
        attacker._probs = snap["probs"]
    elif attacker.name == "SMC":
        attacker._particle_state = snap["particles"]
        attacker._weight_state = snap["weights"]
    elif attacker.name == "ValiditySMC":
        attacker._particle_state = snap["particles"]
        attacker._weight_state = snap["weights"]
        attacker._validity_model.counts.clear()
        for k, v in snap.get("validity_posterior", {}).items():
            attacker._validity_model.counts[k] = [v["alpha"], v["beta"]]

def hash_attacker_state(attacker):
    s = json.dumps(snapshot_attacker_state(attacker), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def evaluate_attacker_state_pure(attacker, samples, defender):
    # Snapshot
    snap = snapshot_attacker_state(attacker)
    initial_hash = hash_attacker_state(attacker)
    
    # Evaluate
    records = attacker.propose(samples, defender)
    
    # Restore
    restore_attacker_state(attacker, snap)
    final_hash = hash_attacker_state(attacker)
    
    assert initial_hash == final_hash, f"State impurity detected in {attacker.name} cross-eval!"
    return records
PY

