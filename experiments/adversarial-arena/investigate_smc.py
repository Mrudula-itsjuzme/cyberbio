import json, os, glob

for seed in ["20260920", "20260921", "20260922"]:
    for r in [0, 1, 2]:
        smc_dir = f"results/arena/carry_reset_smc_paired_seed{seed}/round_{r:03d}"
        vsmc_dir = f"results/arena/carry_reset_validity_smc_paired_seed{seed}/round_{r:03d}"
        
        with open(os.path.join(smc_dir, "reference_evaluation.json")) as f:
            smc_ev = json.load(f)
        with open(os.path.join(vsmc_dir, "reference_evaluation.json")) as f:
            vsmc_ev = json.load(f)
            
        with open(os.path.join(smc_dir, "attacker_state_after.json")) as f:
            smc_sa = json.load(f)
        with open(os.path.join(vsmc_dir, "attacker_state_after.json")) as f:
            vsmc_sa = json.load(f)
            
        print(f"Seed {seed} Round {r}:")
        print("  Carry mean drift eq:", smc_ev["carry_mean_drift"] == vsmc_ev["carry_mean_drift"])
        print("  Carry best drift eq:", smc_ev["carry_best_drift"] == vsmc_ev["carry_best_drift"])
        
        # Check if the states are identical
        import hashlib
        def h(state):
            return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            
        print("  SMC Carry State == VSMC Carry State:", h(smc_sa["carry"]["adaptive_state"]) == h(vsmc_sa["carry"]["adaptive_state"]))
        print("  Validity posterior in VSMC state:", "validity_posterior" in vsmc_sa["carry"]["adaptive_state"])
