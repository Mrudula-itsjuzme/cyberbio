import json, csv
from pathlib import Path
import numpy as np
import pandas as pd

def compute_agg(df, groupby_cols, val_col):
    agg = df.groupby(groupby_cols)[val_col].agg(['mean', 'std']).reset_index()
    # rename
    agg = agg.rename(columns={'mean': f'{val_col}_mean', 'std': f'{val_col}_std'})
    return agg

def main():
    root = Path("results/arena")
    final = Path("results/final")
    
    seeds = ["20260920", "20260921", "20260922"]
    valid_seeds = []
    
    all_rounds = []
    all_cross_attack = []
    all_thompson = []
    
    for seed in seeds:
        run_id = f"arena_transformer_seed{seed}"
        p = root / run_id
        if (p / "run_summary.json").exists():
            with open(p / "run_summary.json") as f:
                d = json.load(f)
                if d.get("status") == "RUN_COMPLETE":
                    valid_seeds.append(seed)
                    
            r = pd.read_csv(p / "final" / "arena_round_summary.csv")
            r['seed'] = seed
            all_rounds.append(r)
            
            ca = pd.read_csv(p / "final" / "arena_cross_attack_matrix.csv")
            ca['seed'] = seed
            all_cross_attack.append(ca)
            
            # Reconstruct thompson
            for round_id in range(len(r)):
                cstate_file = p / f"round_{round_id:03d}" / "controller_state.json"
                if cstate_file.exists():
                    with open(cstate_file) as c_f:
                        ctrl = json.load(c_f)
                    for att in ctrl.get("alpha", {}):
                        all_thompson.append({
                            "seed": seed, "round": round_id, "attacker": att,
                            "alpha": ctrl["alpha"][att], "beta": ctrl["beta"][att],
                            "posterior_mean": ctrl["posterior_means"][att]
                        })

    if not valid_seeds:
        print("No valid seeds found!")
        return
        
    df_rounds = pd.concat(all_rounds, ignore_index=True)
    df_ca = pd.concat(all_cross_attack, ignore_index=True)
    df_th = pd.DataFrame(all_thompson)

    # 1. Multiseed rounds (mean across seeds)
    r_agg = df_rounds.groupby("round").agg({
        "mean_drift": ["mean", "std"],
        "defender_clean_rmse": ["mean", "std"],
        "defender_robustness": ["mean", "std"],
        "defender_gain": ["mean", "std"],
        "validity_rate": ["mean", "std"],
        "candidate_diversity": ["mean", "std"]
    })
    r_agg.columns = ['_'.join(col) for col in r_agg.columns.values]
    r_agg.reset_index().to_csv(final/"arena_multiseed_rounds.csv", index=False)
    
    # 2. Transfer (Held-out Evolutionary)
    # T_Evo(t) = R(Evo, D_t) - R(Evo, D_t+1)
    transfer_records = []
    for seed in valid_seeds:
        ca_s = df_ca[(df_ca["seed"] == seed) & (df_ca["attacker"] == "Evolutionary")]
        for t in range(df_rounds["round"].max()):
            v_dt = ca_s[ca_s["round"] == t-1]["mean_robustness_effect"]
            v_dt1 = ca_s[ca_s["round"] == t]["mean_robustness_effect"]
            if not v_dt.empty and not v_dt1.empty:
                transfer_records.append({"seed": seed, "round": t, "transfer_gain": v_dt.values[0] - v_dt1.values[0]})
    df_trans = pd.DataFrame(transfer_records)
    if not df_trans.empty:
        trans_agg = compute_agg(df_trans, ["round"], "transfer_gain")
        trans_agg.to_csv(final/"arena_multiseed_transfer.csv", index=False)
    else:
        pd.DataFrame(columns=["round", "transfer_gain_mean", "transfer_gain_std"]).to_csv(final/"arena_multiseed_transfer.csv", index=False)

    # 3. Adaptive Selection Gain
    # G_select(t) = R(A_{t+1}, D_t) - R(A_t, D_t)
    gain_records = []
    for seed in valid_seeds:
        r_s = df_rounds[df_rounds["seed"] == seed]
        ca_s = df_ca[df_ca["seed"] == seed]
        for t in range(len(r_s) - 1):
            A_t = r_s[r_s["round"] == t]["active_attacker"].values[0]
            A_t1 = r_s[r_s["round"] == t+1]["active_attacker"].values[0]
            # Matrix evaluating D_t is at round t-1
            row_dt_At1 = ca_s[(ca_s["round"] == t-1) & (ca_s["attacker"] == A_t1)]
            row_dt_At = ca_s[(ca_s["round"] == t-1) & (ca_s["attacker"] == A_t)]
            if not row_dt_At1.empty and not row_dt_At.empty:
                val = row_dt_At1["mean_robustness_effect"].values[0] - row_dt_At["mean_robustness_effect"].values[0]
                gain_records.append({"seed": seed, "round": t, "selection_gain": val})
    df_gain = pd.DataFrame(gain_records)
    if not df_gain.empty:
        gain_agg = compute_agg(df_gain, ["round"], "selection_gain")
        gain_agg.to_csv(final/"arena_multiseed_selection_gain.csv", index=False)

    # 4. Thompson probabilities
    if True:
        if not df_th.empty:
            th_agg = compute_agg(df_th, ["round", "attacker"], "posterior_mean")
        th_agg.to_csv(final/"arena_multiseed_thompson.csv", index=False)

    # 5. Robustness redistribution
    # For each round and seed: Does vulnerability to active attacker decrease, while another increases?
    # R(A, D_t) vs R(A, D_{t+1})  (which is ca[t-1] vs ca[t])
    redist_records = []
    for seed in valid_seeds:
        r_s = df_rounds[df_rounds["seed"] == seed]
        ca_s = df_ca[df_ca["seed"] == seed]
        for t in range(len(r_s)):
            active = r_s[r_s["round"] == t]["active_attacker"].values[0]
            for other in ["RandomSMILES", "MCMC", "Evolutionary"]:
                if other != active:
                    active_before = ca_s[(ca_s["round"] == t-1) & (ca_s["attacker"] == active)]["mean_robustness_effect"]
                    active_after = ca_s[(ca_s["round"] == t) & (ca_s["attacker"] == active)]["mean_robustness_effect"]
                    other_before = ca_s[(ca_s["round"] == t-1) & (ca_s["attacker"] == other)]["mean_robustness_effect"]
                    other_after = ca_s[(ca_s["round"] == t) & (ca_s["attacker"] == other)]["mean_robustness_effect"]
                    
                    if not active_before.empty and not active_after.empty and not other_before.empty and not other_after.empty:
                        ab = active_before.values[0]
                        aa = active_after.values[0]
                        ob = other_before.values[0]
                        oa = other_after.values[0]
                        active_gain = ab - aa
                        other_gain = ob - oa
                        redist = (active_gain > 0) and (other_gain < 0)
                        redist_records.append({
                            "seed": seed, "round": t, "active_attacker": active,
                            "active_before": ab, "active_after": aa, "active_gain": active_gain,
                            "other_attacker": other, "other_before": ob, "other_after": oa, "other_gain": other_gain,
                            "redistribution_observed": redist
                        })
    pd.DataFrame(redist_records).to_csv(final/"arena_robustness_redistribution.csv", index=False)
    
    # 6. Clean vs Robust Tradeoff
    # ROBUST_WIN_CLEAN_WIN, ROBUST_WIN_CLEAN_COST, ROBUST_LOSS_CLEAN_WIN, ROBUST_LOSS_CLEAN_COST
    tradeoff_records = []
    for seed in valid_seeds:
        r_s = df_rounds[df_rounds["seed"] == seed]
        for t in range(len(r_s) - 1): # comparing t and t+1
            c_t = r_s[r_s["round"] == t]["defender_clean_rmse"].values[0]
            c_t1 = r_s[r_s["round"] == t+1]["defender_clean_rmse"].values[0]
            rob_t = r_s[r_s["round"] == t]["defender_robustness"].values[0]
            rob_t1 = r_s[r_s["round"] == t+1]["defender_robustness"].values[0]
            
            delta_clean = c_t1 - c_t
            delta_robust = rob_t - rob_t1 # positive if robust effect decreased (win)
            
            clean_win = delta_clean <= 0
            robust_win = delta_robust > 0
            
            if robust_win and clean_win: cls = "ROBUST_WIN_CLEAN_WIN"
            elif robust_win and not clean_win: cls = "ROBUST_WIN_CLEAN_COST"
            elif not robust_win and clean_win: cls = "ROBUST_LOSS_CLEAN_WIN"
            else: cls = "ROBUST_LOSS_CLEAN_COST"
            
            tradeoff_records.append({
                "seed": seed, "round": t, "delta_clean": delta_clean, "delta_robust": delta_robust,
                "classification": cls
            })
    pd.DataFrame(tradeoff_records).to_csv(final/"arena_clean_robust_tradeoff.csv", index=False)
    
    # Save seed validation
    with open("docs/ARENA_SEED_VALIDATION.md", "w") as f:
        f.write("# Arena Seed Validation\n\n")
        for s in seeds:
            f.write(f"## Seed {s}\nStatus: {'VALID' if s in valid_seeds else 'FAILED'}\n")

if __name__ == "__main__":
    main()
