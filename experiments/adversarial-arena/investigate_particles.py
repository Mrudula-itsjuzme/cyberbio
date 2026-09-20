import json

with open("results/arena/carry_reset_smc_paired_seed20260920/round_000/attacker_state_after.json") as f:
    smc_sa = json.load(f)
with open("results/arena/carry_reset_validity_smc_paired_seed20260920/round_000/attacker_state_after.json") as f:
    vsmc_sa = json.load(f)
    
print("SMC particles == VSMC particles:", smc_sa["carry"]["adaptive_state"]["particle_state"] == vsmc_sa["carry"]["adaptive_state"]["particle_state"])
