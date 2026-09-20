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
