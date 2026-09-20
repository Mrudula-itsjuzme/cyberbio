import json

with open("results/arena/carry_reset_smc_paired_seed20260920/round_000/attacker_state_after.json") as f:
    smc_sa = json.load(f)
    
print(smc_sa["carry"]["adaptive_state"].keys())
