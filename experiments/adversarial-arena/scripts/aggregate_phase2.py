import json, csv, sys, shutil, math
from pathlib import Path
from collections import defaultdict
import numpy as np

def require_float(val, name, context):
    if val is None or val == "":
        raise ValueError(f"Missing required float for {name} in {context}")
    return float(val)

def optional_float(val):
    if val is None or val == "" or str(val).lower() == "none":
        return np.nan
    return float(val)

def aggregate():
    res_dir = Path("results/arena")
    out_dir = Path("results/final/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    runs = list(res_dir.glob("phase2_*_carry_seed*"))
    if len(runs) != 9:
        print(f"FAILED: Expected 9 runs, found {len(runs)}")
        sys.exit(1)
        
    rounds_data, attackers_data, transfer_data = [], [], []
    defenders_data, redistribution_data, tradeoff_data = [], [], []
    diversity_data, summary_data = [], []
    dynamics_data = {"cem": [], "smc": [], "validity_smc": []}
    
    g_adapt_raw = []
    
    for rdir in sorted(runs):
        parts = rdir.name.split("_")
        method = "validity_smc" if "validity" in parts else parts[1]
        seed = parts[-1].replace("seed", "")
        
        sum_file = rdir / "run_summary.json"
        with open(sum_file) as f:
            summary = json.load(f)
            
        agg_mean_drift = []
        agg_best_drift = []
        agg_def_gain_diff = []
        agg_t_evo = []
        redistribution_count = 0
        total_queries = 0
        val_rate_sum = 0
        wall_clock = 0
        
        # Get D0 clean RMSE
        clean_rmse_D0 = None
        rd0 = rdir / "round_000"
        def_m_file0 = rd0 / "defender_metrics.json"
        if def_m_file0.exists():
            clean_rmse_D0 = require_float(json.load(open(def_m_file0)).get("rmse"), "rmse", f"{rdir.name}/round_000")
            
        last_clean_rmse = None
        
        for round_idx, round_info in enumerate(summary.get("rounds", [])):
            rd = rdir / f"round_{round_idx:03d}"
            att_file = rd / "attacker_state_after.json"
            att_data = {}
            if att_file.exists():
                with open(att_file) as af: att_data = json.load(af)
                
            baseline_csv = rd / "baseline_cross_attack_matrix.csv"
            if not baseline_csv.exists() and round_idx > 0:
                baseline_csv = rdir / f"round_{(round_idx-1):03d}" / "cross_attack_matrix.csv"
                
            after_csv = rd / "cross_attack_matrix.csv"
            
            b_data = {}
            if baseline_csv.exists():
                with open(baseline_csv) as f:
                    for row in csv.DictReader(f):
                        b_data[row["attacker"]] = optional_float(row["mean_drift"])
            
            a_data = {}
            if after_csv.exists():
                with open(after_csv) as f:
                    for row in csv.DictReader(f):
                        a_data[row["attacker"]] = optional_float(row["mean_drift"])
            
            active_attacker_key = None
            for key in a_data.keys():
                if method.lower() in key.lower():
                    active_attacker_key = key
                    break
            if not active_attacker_key:
                active_attacker_key = method
            
            if active_attacker_key not in b_data or active_attacker_key not in a_data:
                raise ValueError(f"Active attacker {active_attacker_key} not found in cross attack matrix for {rdir.name}/round_{round_idx:03d}")
                
            active_drift_before = b_data[active_attacker_key]
            active_drift_after = a_data[active_attacker_key]
            active_gain_difference = active_drift_before - active_drift_after
            
            def_m_file = rd / "defender_metrics.json"
            if def_m_file.exists():
                clean_after = require_float(json.load(open(def_m_file)).get("rmse"), "rmse", f"{rdir.name}/round_{round_idx:03d}")
            else:
                clean_after = clean_rmse_D0
                
            if round_idx == 0:
                clean_before = clean_rmse_D0
            else:
                clean_before = last_clean_rmse
            last_clean_rmse = clean_after
            
            delta_clean_round = clean_after - clean_before
            
            def_gain_ratio = optional_float(round_info.get("defender_gain"))
            
            rounds_data.append({
                "method": method, "seed": seed, "round": round_idx,
                "clean_rmse_before": clean_before,
                "clean_rmse_after": clean_after,
                "delta_clean": delta_clean_round,
                "active_gain_difference": active_gain_difference,
                "hard_examples": round_info.get("hard_example_buffer_size", 0)
            })
            
            # attackers
            total_proposals = require_float(round_info.get("total_proposals", att_data.get("proposal_count")), "proposal_count", f"{rdir.name}/round_{round_idx:03d}")
            best_drift = require_float(att_data.get("best_drift", round_info.get("best_drift")), "best_drift", f"{rdir.name}/round_{round_idx:03d}")
            mean_drift = require_float(att_data.get("mean_drift", round_info.get("mean_drift")), "mean_drift", f"{rdir.name}/round_{round_idx:03d}")
            queries = require_float(round_info.get("target_model_queries"), "target_queries", f"{rdir.name}/round_{round_idx:03d}")
            valid_c = require_float(round_info.get("valid_count"), "valid_count", f"{rdir.name}/round_{round_idx:03d}")
            rt = optional_float(att_data.get("runtime_seconds"))
            attackers_data.append({
                "method": method, "seed": seed, "round": round_idx,
                "valid_count": valid_c,
                "target_queries": queries,
                "proposals": total_proposals,
                "best_drift": best_drift,
                "mean_drift": mean_drift,
                "runtime": rt
            })
            
            # dynamics
            d_row = {"method": method, "seed": seed, "round": round_idx}
            if method == "cem":
                h = optional_float(att_data.get("entropy"))
                h_norm = h / math.log(10) if not np.isnan(h) else np.nan # assuming 10 edits
                d_row["entropy"] = h
                d_row["normalized_entropy"] = h_norm
                d_row["probs"] = str(att_data.get("probs", []))
            elif method == "smc" or method == "validity_smc":
                ess = optional_float(att_data.get("ess"))
                diversity = optional_float(att_data.get("diversity"))
                n_particles = 100 # Default/Assuming 100
                d_row["ess"] = ess
                d_row["ess_ratio"] = ess / n_particles if not np.isnan(ess) else np.nan
                d_row["diversity"] = diversity
                if method == "validity_smc":
                    d_row["beta_a"] = optional_float(att_data.get("beta_a"))
                    d_row["beta_b"] = optional_float(att_data.get("beta_b"))
            dynamics_data[method].append(d_row)

            # defenders
            defenders_data.append({
                "method": method, "seed": seed, "round": round_idx,
                "checkpoint_before": f"round_{round_idx-1}" if round_idx > 0 else "initial",
                "checkpoint_after": f"round_{round_idx}",
                "clean_rmse_before": clean_before,
                "clean_rmse_after": clean_after,
                "delta_clean": delta_clean_round,
                "active_attacker": active_attacker_key,
                "active_drift_before": active_drift_before,
                "active_drift_after": active_drift_after,
                "defender_gain_difference": active_gain_difference,
                "defender_gain_ratio": def_gain_ratio
            })

            # transfer
            evo_key = "Evolutionary" if "Evolutionary" in b_data else None
            if evo_key:
                evo_before = b_data[evo_key]
                evo_after = a_data[evo_key]
                t_evo = evo_before - evo_after
                transfer_data.append({
                    "method": method, "seed": seed, "round": round_idx,
                    "R_before": evo_before,
                    "R_after": evo_after,
                    "T_Evo": t_evo
                })
                agg_t_evo.append(t_evo)
                
            # redistribution
            if evo_key:
                other_gain_diff = evo_before - evo_after
                redistribution_observed = (active_gain_difference > 0) and (other_gain_diff < 0)
                if redistribution_observed: redistribution_count += 1
                redistribution_data.append({
                    "method": method, "seed": seed, "round": round_idx,
                    "active_attacker": active_attacker_key, "other_attacker": evo_key,
                    "active_before": active_drift_before, "active_after": active_drift_after,
                    "active_gain": active_gain_difference, 
                    "other_before": evo_before, "other_after": evo_after,
                    "other_gain": other_gain_diff,
                    "redistribution_observed": redistribution_observed
                })
                
            # tradeoff
            delta_clean = clean_after - clean_before
            delta_robust = active_gain_difference
            if delta_robust > 0 and delta_clean <= 0: classification = "ROBUST_WIN_CLEAN_WIN"
            elif delta_robust > 0 and delta_clean > 0: classification = "ROBUST_WIN_CLEAN_COST"
            elif delta_robust <= 0 and delta_clean <= 0: classification = "ROBUST_LOSS_CLEAN_WIN"
            else: classification = "ROBUST_LOSS_CLEAN_COST"
            tradeoff_data.append({
                "method": method, "seed": seed, "round": round_idx,
                "delta_clean": delta_clean,
                "delta_robust": delta_robust,
                "classification": classification
            })
            
            # diversity (candidates)
            cands_file = rd / "candidates.jsonl"
            uniq_c = 0
            dup_frac = 0.0
            if cands_file.exists():
                cands = [json.loads(line) for line in cands_file.read_text().splitlines() if line.strip()]
                unique_smiles = set(c.get("candidate_input", "") for c in cands)
                uniq_c = len(unique_smiles)
                dup_frac = 1.0 - (uniq_c / len(cands)) if len(cands) > 0 else 0.0
            diversity_data.append({
                "method": method, "seed": seed, "round": round_idx,
                "unique_candidate_count": uniq_c,
                "duplicate_fraction": dup_frac
            })
            
            # agg
            agg_mean_drift.append(mean_drift)
            agg_best_drift.append(best_drift)
            agg_def_gain_diff.append(active_gain_difference)
            total_queries += queries
            val_rate_sum += (valid_c / total_proposals) if total_proposals > 0 else 0
            if not np.isnan(rt): wall_clock += rt
            
        delta_clean_total = last_clean_rmse - clean_rmse_D0 if clean_rmse_D0 is not None and last_clean_rmse is not None else np.nan
                
        # load G_adapt to include in summary
        g_adapt_mean = np.nan
        raw_g = res_dir / "arena_phase2_g_adapt_raw.csv"
        if raw_g.exists():
            with open(raw_g) as gf:
                g_vals = []
                for r in csv.DictReader(gf):
                    if r["run"] == rdir.name:
                        g_vals.append(require_float(r["G_adapt"], "G_adapt", "raw G_adapt"))
                        g_adapt_raw.append({
                            "method": method,
                            "seed": seed,
                            "round": r["round"],
                            "R_before": r["R_At_Dt"],
                            "R_after": r["R_At1_Dt"],
                            "G_adapt": r["G_adapt"],
                            "query_budget": 250
                        })
                if g_vals: g_adapt_mean = np.mean(g_vals)
                
        summary_data.append({
            "method": method, "seed": seed,
            "mean_drift": np.mean(agg_mean_drift) if agg_mean_drift else np.nan,
            "best_drift": np.max(agg_best_drift) if agg_best_drift else np.nan,
            "validity_rate": val_rate_sum / len(summary.get("rounds", [])),
            "target_queries": total_queries,
            "G_adapt_mean": g_adapt_mean,
            "defender_gain_diff_mean": np.mean(agg_def_gain_diff) if agg_def_gain_diff else np.nan,
            "T_Evo_mean": np.mean(agg_t_evo) if agg_t_evo else np.nan,
            "redistribution_count": redistribution_count,
            "clean_rmse_delta_total": delta_clean_total,
            "runtime": wall_clock
        })

    def write_csv(name, data):
        if not data: return
        with open(out_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            
    write_csv("phase2_rounds.csv", rounds_data)
    write_csv("phase2_attackers.csv", attackers_data)
    write_csv("phase2_cem_dynamics.csv", dynamics_data["cem"])
    write_csv("phase2_smc_dynamics.csv", dynamics_data["smc"])
    write_csv("phase2_validity_smc_dynamics.csv", dynamics_data["validity_smc"])
    write_csv("phase2_defenders.csv", defenders_data)
    write_csv("phase2_transfer.csv", transfer_data)
    write_csv("phase2_redistribution.csv", redistribution_data)
    write_csv("phase2_clean_robust_tradeoff.csv", tradeoff_data)
    write_csv("phase2_diversity.csv", diversity_data)
    write_csv("phase2_summary.csv", summary_data)
    write_csv("phase2_g_adapt.csv", g_adapt_raw)
    
    # Generate Phase 2 Statistics
    stats = []
    
    # Helper to calculate stats
    def calc_stats(metric_name, data, value_key, method_key="method"):
        method_groups = defaultdict(list)
        for d in data:
            val = optional_float(d.get(value_key))
            if not np.isnan(val):
                method_groups[d[method_key]].append(val)
                
        for meth, vals in method_groups.items():
            stats.append({
                "metric": metric_name, "method": meth, "round": "all",
                "n": len(vals), "mean": np.mean(vals), "sample_std": np.std(vals, ddof=1) if len(vals)>1 else 0,
                "median": np.median(vals), "min": np.min(vals), "max": np.max(vals)
            })
            
    calc_stats("mean_drift", attackers_data, "mean_drift")
    calc_stats("best_drift", attackers_data, "best_drift")
    calc_stats("G_adapt", g_adapt_raw, "G_adapt")
    calc_stats("defender_gain_difference", defenders_data, "defender_gain_difference")
    calc_stats("T_Evo", transfer_data, "T_Evo")
    calc_stats("ESS_ratio", dynamics_data["smc"], "ess_ratio")
    calc_stats("ESS_ratio", dynamics_data["validity_smc"], "ess_ratio")
    calc_stats("normalized_entropy", dynamics_data["cem"], "normalized_entropy")
    
    write_csv("phase2_statistics.csv", stats)

    print("Aggregation logic complete.")

if __name__ == "__main__": aggregate()
