import pandas as pd
import os

os.makedirs("docs", exist_ok=True)

df = pd.read_csv("results/final/carry_reset/carry_reset_events.csv")
multiseed = pd.read_csv("results/final/carry_reset/carry_reset_multiseed_statistics.csv")

with open("docs/CARRY_RESET_RESULTS.md", "w") as f:
    f.write("# Phase 2B Carry/Reset Results\n\n")
    f.write("This document summarizes the G_carry memory effect.\n\n")
    f.write("## Overall Performance\n")
    f.write(multiseed.to_csv(index=False))

with open("docs/CARRY_RESET_RQ_LEDGER.md", "w") as f:
    f.write("# RQ Ledger\n\n")
    f.write("## Primary: Does preserving attacker memory across rounds improve adversarial search?\n")
    # simplistic logic for verdict based on median of medians
    med = multiseed["median_G_carry_mean"].median()
    if med > 0.05:
        verdict = "SUPPORTED_IN_PILOT"
    elif med < -0.05:
        verdict = "NOT_SUPPORTED_IN_PILOT"
    else:
        verdict = "MIXED"
    f.write(f"**Verdict**: {verdict}\n")

with open("docs/CARRY_RESET_NEGATIVE_RESULTS.md", "w") as f:
    f.write("# Negative Results\n\n")
    neg = df[df["G_carry_mean"] <= 0]
    f.write(f"Found {len(neg)} events where G_carry <= 0.\n\n")
    f.write(neg[["method", "seed", "round", "G_carry_mean"]].to_csv(index=False))

with open("docs/CARRY_RESET_VIVA.md", "w") as f:
    f.write("# Viva Notes\n\n")
    f.write("G_adapt measures within-attacker adaptation.\n")
    f.write("G_carry measures persistent-memory vs reset contrast under a shared D_ref.\n")

with open("docs/CARRY_RESET_RUN_VALIDATION.md", "w") as f:
    f.write("# Run Validation\n\n")
    invariants = pd.read_csv("results/final/carry_reset/carry_reset_state_invariants.csv")
    f.write(f"All invariant checks: {invariants['carry_invariant_pass'].all() and invariants['reset_invariant_pass'].all() and invariants['rng_invariant_pass'].all()}\n")
