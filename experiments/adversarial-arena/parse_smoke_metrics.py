import json, os, glob

for method in ["cem", "smc", "validity_smc"]:
    d = f"results/arena/smoke_carry_reset_{method}_paired_seed20260920"
    for r in range(2):
        with open(os.path.join(d, f"round_{r:03d}", "reference_evaluation.json")) as f:
            ev = json.load(f)
        with open(os.path.join(d, f"round_{r:03d}", "attacker_state_before.json")) as f:
            sb = json.load(f)
        with open(os.path.join(d, f"round_{r:03d}", "attacker_state_after.json")) as f:
            sa = json.load(f)
            
        print(f"=== {method} Round {r} ===")
        print(f"Ref ID: {ev.get('reference_checkpoint_id', 'canonical-D0')}")
        print(f"Ref SHA256: {ev['reference_checkpoint_sha256']}")
        print(f"Carry queries: {ev['carry_target_queries']} | Reset queries: {ev['reset_target_queries']} | Delta: {ev['delta_queries']}")
        print(f"Carry Mean Drift: {ev['carry_mean_drift']} | Reset Mean Drift: {ev['reset_mean_drift']} | G_carry_mean: {ev['G_carry_mean']}")
        print(f"Carry Best Drift: {ev['carry_best_drift']} | Reset Best Drift: {ev['reset_best_drift']} | G_carry_best: {ev['G_carry_best']}")
        
        import hashlib
        def h(state): return hashlib.sha256(json.dumps(state['adaptive_state'], sort_keys=True).encode()).hexdigest()[:8]
        
        print(f"Carry hash: {h(sb['carry'])} -> {h(sa['carry'])}")
        print(f"Reset hash: {h(sb['reset'])} -> {h(sa['reset'])}")
        print()
