import json

for seed in ["20260920", "20260921", "20260922"]:
    for r in [0,1,2]:
        try:
            with open(f"results/arena/carry_reset_smc_paired_seed{seed}/round_{r:03d}/attacker_state_after.json") as f:
                s1 = json.load(f)["carry"]["adaptive_state"]["particles"]
            with open(f"results/arena/carry_reset_validity_smc_paired_seed{seed}/round_{r:03d}/attacker_state_after.json") as f:
                s2 = json.load(f)["carry"]["adaptive_state"]["particles"]
            print(f"Seed {seed} Round {r}: particles equal? {s1 == s2}")
        except:
            pass
