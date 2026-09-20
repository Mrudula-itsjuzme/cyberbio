import json, os

def get_smoke_dir(method):
    return f"results/arena/smoke_carry_reset_{method}_paired_seed20260920"

def test_cem_reset_state():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "round_000", "attacker_state_before.json")) as f:
        r0 = json.load(f)["reset"]
    with open(os.path.join(d, "round_001", "attacker_state_before.json")) as f:
        r1 = json.load(f)["reset"]
    assert r0["adaptive_state"] == r1["adaptive_state"]

def test_smc_reset_state():
    d = get_smoke_dir("smc")
    with open(os.path.join(d, "round_000", "attacker_state_before.json")) as f:
        r0 = json.load(f)["reset"]
    with open(os.path.join(d, "round_001", "attacker_state_before.json")) as f:
        r1 = json.load(f)["reset"]
    assert r0["adaptive_state"] == r1["adaptive_state"]

def test_validity_smc_reset_state():
    d = get_smoke_dir("validity_smc")
    with open(os.path.join(d, "round_000", "attacker_state_before.json")) as f:
        r0 = json.load(f)["reset"]
    with open(os.path.join(d, "round_001", "attacker_state_before.json")) as f:
        r1 = json.load(f)["reset"]
    assert r0["adaptive_state"] == r1["adaptive_state"]

def test_cem_carry_hash_chain():
    d = get_smoke_dir("cem")
    with open(os.path.join(d, "round_000", "attacker_state_after.json")) as f:
        r0a = json.load(f)["carry"]
    with open(os.path.join(d, "round_001", "attacker_state_before.json")) as f:
        r1b = json.load(f)["carry"]
    assert r0a["adaptive_state"] == r1b["adaptive_state"]

def test_smc_carry_hash_chain():
    d = get_smoke_dir("smc")
    with open(os.path.join(d, "round_000", "attacker_state_after.json")) as f:
        r0a = json.load(f)["carry"]
    with open(os.path.join(d, "round_001", "attacker_state_before.json")) as f:
        r1b = json.load(f)["carry"]
    assert r0a["adaptive_state"] == r1b["adaptive_state"]

def test_validity_smc_carry_hash_chain():
    d = get_smoke_dir("validity_smc")
    with open(os.path.join(d, "round_000", "attacker_state_after.json")) as f:
        r0a = json.load(f)["carry"]
    with open(os.path.join(d, "round_001", "attacker_state_before.json")) as f:
        r1b = json.load(f)["carry"]
    assert r0a["adaptive_state"] == r1b["adaptive_state"]
