import json

for seed in ["20260920", "20260921", "20260922"]:
    with open(f"results/arena/carry_reset_smc_paired_seed{seed}/round_000/history.json") as f:
        hist = json.load(f)
        for h in hist["carry"]:
            if h.get("resampled"):
                print(f"Seed {seed} resampled in SMC!")
