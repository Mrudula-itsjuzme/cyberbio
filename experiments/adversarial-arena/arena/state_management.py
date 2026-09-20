import json
import hashlib
import copy

def derive_seed(base_seed, round_id, source_id, role):
    s = f"{base_seed}_{round_id}_{source_id}_{role}"
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**32)

def get_adaptive_state(attacker):
    state = {}
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

def snapshot_attacker_state(attacker):
    return {
        "round_id": getattr(attacker, "_round_id", 0),
        "adaptive_state": get_adaptive_state(attacker)
    }

def restore_attacker_state(attacker, snap):
    attacker._round_id = snap.get("round_id", 0)
    astate = snap["adaptive_state"]
    if attacker.name == "CEM":
        attacker._probs = astate["probs"]
    elif attacker.name == "SMC":
        attacker._particle_state = astate["particles"]
        attacker._weight_state = astate["weights"]
    elif attacker.name == "ValiditySMC":
        attacker._particle_state = astate["particles"]
        attacker._weight_state = astate["weights"]
        attacker._validity_model.counts.clear()
        for k, v in astate.get("validity_posterior", {}).items():
            attacker._validity_model.counts[k] = [v["alpha"], v["beta"]]

def hash_adaptive_state(attacker):
    s = json.dumps(get_adaptive_state(attacker), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def hash_full_state(attacker):
    s = json.dumps(snapshot_attacker_state(attacker), sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()

def evaluate_attacker_state_pure(attacker, samples, defender):
    snap = snapshot_attacker_state(attacker)
    initial_full_hash = hash_full_state(attacker)
    
    temp_att = copy.deepcopy(attacker)
    records = temp_att.propose(samples, defender)
    metrics = temp_att.get_metrics()
    
    restore_attacker_state(attacker, snap)
    final_full_hash = hash_full_state(attacker)
    
    assert initial_full_hash == final_full_hash, "State purity violation!"
    
    return {
        "records": records,
        "metrics": metrics,
        "adaptive_state_after_eval": get_adaptive_state(temp_att)
    }
