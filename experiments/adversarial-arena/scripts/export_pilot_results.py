import json, csv, shutil
from pathlib import Path

def main():
    root = Path("results/arena/test_pilot")
    final = Path("results/final")
    final.mkdir(parents=True, exist_ok=True)
    
    with open(root/"run_summary.json") as f:
        run_data = json.load(f)
        
    shutil.copy(root/"final"/"arena_round_summary.csv", final/"arena_pilot_rounds.csv")
    shutil.copy(root/"final"/"arena_attacker_dynamics.csv", final/"arena_pilot_attackers.csv")
    shutil.copy(root/"final"/"arena_defender_dynamics.csv", final/"arena_pilot_defenders.csv")
    shutil.copy(root/"final"/"arena_cross_attack_matrix.csv", final/"arena_pilot_cross_attack.csv")
    shutil.copy(root/"final"/"arena_hard_example_stats.csv", final/"arena_pilot_hard_examples.csv")
    
    # Thompson
    with open(final/"arena_pilot_thompson.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "attacker", "alpha", "beta", "posterior_mean"])
        for round_id in range(len(run_data["rounds"])):
            cstate_file = root / f"round_{round_id}" / "controller_state.json"
            if cstate_file.exists():
                with open(cstate_file) as c_f:
                    ctrl = json.load(c_f)
                for att in ctrl.get("alpha", {}):
                    writer.writerow([round_id, att, ctrl["alpha"][att], ctrl["beta"][att], ctrl["posterior_means"][att]])
                    
    # Metrics
    # T_j(t) = R(A_j, D_t) - R(A_j, D_{t+1})
    # G_select(t) = R(A_{t+1}, D_t) - R(A_t, D_t)
    
    ca = list(csv.DictReader(open(final/"arena_pilot_cross_attack.csv")))
    # Group by round and attacker
    # D_0 evals are saved as round -1
    # D_{t+1} evals are saved as round t
    
    # We will build a dict: D_evals[d_index][attacker] = mean_drift
    D_evals = {}
    for row in ca:
        r = int(row["round"])
        d_idx = r + 1 # so round -1 (D0) is 0, round 0 (D1) is 1
        att = row["attacker"]
        val = float(row["mean_robustness_effect"])
        if d_idx not in D_evals: D_evals[d_idx] = {}
        D_evals[d_idx][att] = val

    # We need to know which attacker was active at each round.
    rounds_data = list(csv.DictReader(open(final/"arena_pilot_rounds.csv")))
    active_attackers = {}
    for row in rounds_data:
        active_attackers[int(row["round"])] = row["active_attacker"]
        
    with open(final/"arena_pilot_transfer.csv", "w", newline="") as f_t, \
         open(final/"arena_pilot_selection_gain.csv", "w", newline="") as f_a:
        writer_t = csv.writer(f_t)
        writer_t.writerow(["round", "attacker", "transfer_gain"])
        
        writer_a = csv.writer(f_a)
        writer_a.writerow(["round", "attacker_t", "attacker_t_plus_1", "adaptation_gain"])

        for t in range(len(rounds_data)):
            # Unseen attackers: all attackers except active_attackers[t]
            for att in D_evals[t].keys():
                if att != active_attackers[t]:
                    # T_j(t) = R(A_j, D_t) - R(A_j, D_{t+1})
                    tj = D_evals[t][att] - D_evals[t+1][att]
                    if att == "Evolutionary":
                        writer_t.writerow([t, att, tj])
                    
            if t < len(rounds_data) - 1:
                A_t = active_attackers[t]
                A_t_next = active_attackers[t+1]
                # G_select(t) = R(A_{t+1}, D_t) - R(A_t, D_t)
                ga = D_evals[t][A_t_next] - D_evals[t][A_t]
                writer_a.writerow([t, A_t, A_t_next, ga])

if __name__ == "__main__": main()
