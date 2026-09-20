import json

with open("results/arena/carry_reset_smc_paired_seed20260920/round_000/reference_evaluation.json") as f:
    smc_ev = json.load(f)
    
print("SMC ESS:", smc_ev.get("ess", "Not found"))

with open("results/arena/carry_reset_smc_paired_seed20260920/round_000/attacker_state_after.json") as f:
    smc_sa = json.load(f)
print("ESS in state_after:", smc_sa["carry"]["metrics"]["ess"])
