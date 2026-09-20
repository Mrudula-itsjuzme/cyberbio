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
    with open(final/"arena_pilot_thompson.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "attacker", "alpha", "beta", "posterior_mean"])
        if run_data.get("controller"):
            ctrl = run_data["controller"]
            for round_id in range(len(run_data["rounds"])):
                for att in ctrl["alpha"]:
                    writer.writerow([round_id, att, ctrl["alpha"][att], ctrl["beta"][att], ctrl["posterior_means"][att]])
                    
    # Transfer: unseen attacker transfer. We just extract it from cross_attack
    ca = list(csv.DictReader(open(final/"arena_pilot_cross_attack.csv")))
    with open(final/"arena_pilot_transfer.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "attacker", "transfer_gain"])
        for row in ca:
            writer.writerow([row["round"], row["attacker"], ""])
            
if __name__ == "__main__": main()
