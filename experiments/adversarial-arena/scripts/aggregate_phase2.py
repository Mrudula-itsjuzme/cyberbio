import json, csv, sys, shutil, math
from pathlib import Path
from collections import defaultdict
import numpy as np

def require_float(val, name, context):
    if val is None or val == "" or str(val).lower() == "none":
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
    
    ACTIVE_LABEL = {
        "cem": "cem",
        "smc": "smc",
        "validity_smc": "validity_smc"
    }
    
    method_cands = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    method_hard_cands = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    val_eff = defaultdict(lambda: defaultdict(dict))
    
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
            
            active_attacker_key = ACTIVE_LABEL.get(method)
            if not active_attacker_key or active_attacker_key not in b_data or active_attacker_key not in a_data:
                print(f"FAILED: Exact mapping required. {method} not strictly matched in {rdir.name}/round_{round_idx:03d}")
                sys.exit(1)
                
            with open("docs/ARENA_PHASE2_LABEL_MAPPING.md", "a") as mf:
                mf.write(f"- EXACT MAPPING VERIFIED: `{method}` -> `{active_attacker_key}` in {rdir.name}/round_{round_idx:03d}\n")
                
            active_drift_before = b_data[active_attacker_key]
            active_drift_after = a_data[active_attacker_key]
            active_gain_difference = active_drift_before - active_drift_after if not np.isnan(active_drift_before) else np.nan
            
            # clean RMSE logic (provenance fix: it's D_{t+1}, so clean_before is D_{t})
            def_m_file = rd / "defender_metrics.json"
            if def_m_file.exists():
                clean_after = require_float(json.load(open(def_m_file)).get("rmse"), "rmse", f"{rdir.name}/round_{round_idx:03d}")
            else:
                clean_after = np.nan
                
            if round_idx == 0:
                clean_before = np.nan # Initial D0 clean RMSE is lost/never persisted
            else:
                clean_before = last_clean_rmse
            last_clean_rmse = clean_after
            
            delta_clean_round = clean_after - clean_before if clean_before is not np.nan else np.nan
            def_gain_ratio = optional_float(round_info.get("defender_gain"))
            
            rounds_data.append({
                "method": method, "seed": seed, "round": round_idx,
                "clean_rmse_before": clean_before,
                "clean_rmse_after": clean_after,
                "delta_clean": delta_clean_round,
                "active_gain_difference": active_gain_difference,
                "hard_examples": round_info.get("hard_example_buffer_size", 0)
            })
            
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
            
            cands_file = rd / "candidates.jsonl"
            uniq_c = 0
            dup_frac = 0.0
            unique_smiles = set()
            if cands_file.exists():
                cands = [json.loads(line) for line in cands_file.read_text().splitlines() if line.strip()]
                unique_smiles = set(c.get("candidate_input") for c in cands if c.get("candidate_input"))
                uniq_c = len(unique_smiles)
                dup_frac = 1.0 - (uniq_c / len(cands)) if len(cands) > 0 else 0.0
            
            method_cands[seed][round_idx][method] = unique_smiles
            
            hard_file = rd / "hard_examples.jsonl"
            hard_smiles = set()
            if hard_file.exists():
                hcands = [json.loads(line) for line in hard_file.read_text().splitlines() if line.strip()]
                hard_smiles = set(c.get("candidate_input") for c in hcands if c.get("candidate_input"))
            method_hard_cands[seed][round_idx][method] = hard_smiles
            
            diversity_data.append({
                "method": method, "seed": seed, "round": round_idx,
                "unique_candidate_count": uniq_c,
                "duplicate_fraction": dup_frac
            })
            
            d_row = {"method": method, "seed": seed, "round": round_idx}
            if method == "cem":
                h = optional_float(att_data.get("entropy"))
                family_names = att_data.get("state_after", {}).get("family_names")
                if not family_names: family_names = att_data.get("probs", [])
                
                K = len(family_names)
                if K < 2: raise ValueError(f"CEM family count K={K} must be >= 2 for seed {seed} round {round_idx}")
                
                h_max = math.log(K)
                h_norm = h / h_max if not np.isnan(h) else np.nan
                
                if h_norm < 0.20: collapse_cls = "severe collapse"
                elif h_norm < 0.50: collapse_cls = "concentration"
                else: collapse_cls = "broad"
                
                max_prob = max(att_data.get("probs", [0])) if att_data.get("probs") else np.nan
                d_row.update({
                    "family_count": K,
                    "entropy": h,
                    "max_entropy": h_max,
                    "normalized_entropy": h_norm,
                    "dominant_probability": max_prob,
                    "classification": collapse_cls
                })
                dynamics_data["cem"].append(d_row)
            
            elif method == "smc" or method == "validity_smc":
                ess = optional_float(att_data.get("ess"))
                diversity = optional_float(att_data.get("diversity"))
                n_particles = att_data.get("particle_count")
                if n_particles is None: n_particles = att_data.get("state_after", {}).get("particle_count")
                
                if n_particles is None:
                    raise ValueError(f"Missing SMC particle_count for seed {seed} round {round_idx}")
                    
                ess_ratio = ess / float(n_particles) if not np.isnan(ess) else np.nan
                if ess_ratio < 0.20: smc_cls = "severe"
                elif ess_ratio < 0.50: smc_cls = "moderate"
                else: smc_cls = "healthy"
                
                d_row.update({
                    "particle_count": n_particles,
                    "ess": ess,
                    "ess_ratio": ess_ratio,
                    "resampling_count": np.nan, # Not explicit in states
                    "diversity": diversity,
                    "classification": smc_cls
                })
                if method == "validity_smc":
                    d_row["beta_a"] = optional_float(att_data.get("beta_a"))
                    d_row["beta_b"] = optional_float(att_data.get("beta_b"))
                dynamics_data[method].append(d_row)
                
                drift_per_query = mean_drift / queries if queries > 0 else 0
                val_per_query = valid_c / queries if queries > 0 else 0
                val_eff[seed][round_idx][method] = {
                    "proposal_count": total_proposals,
                    "valid_count": valid_c,
                    "target_queries": queries,
                    "validity_rate": valid_c / total_proposals if total_proposals > 0 else 0,
                    "valid_per_target_query": val_per_query,
                    "mean_drift": mean_drift,
                    "best_drift": best_drift,
                    "drift_per_target_query": drift_per_query,
                    "particle_count": n_particles,
                    "ess": ess,
                    "ess_ratio": ess_ratio,
                    "unique_candidates": uniq_c,
                    "duplicate_fraction": dup_frac
                }

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

            evo_key = "Evolutionary" if "Evolutionary" in b_data else None
            if evo_key:
                evo_before = b_data[evo_key]
                evo_after = a_data[evo_key]
                t_evo = evo_before - evo_after if not np.isnan(evo_before) and not np.isnan(evo_after) else np.nan
                transfer_data.append({
                    "method": method, "seed": seed, "round": round_idx,
                    "R_before": evo_before,
                    "R_after": evo_after,
                    "T_Evo": t_evo
                })
                if not np.isnan(t_evo): agg_t_evo.append(t_evo)
                
                other_gain_diff = t_evo
                redistribution_observed = (active_gain_difference > 0) and (other_gain_diff < 0) if not np.isnan(active_gain_difference) else False
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
                
            delta_clean = clean_after - clean_before if clean_before is not np.nan else np.nan
            delta_robust = active_gain_difference
            classification = "NA"
            if not np.isnan(delta_clean) and not np.isnan(delta_robust):
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
            
            agg_mean_drift.append(mean_drift)
            agg_best_drift.append(best_drift)
            if not np.isnan(active_gain_difference): agg_def_gain_diff.append(active_gain_difference)
            total_queries += queries
            val_rate_sum += (valid_c / total_proposals) if total_proposals > 0 else 0
            if not np.isnan(rt): wall_clock += rt
            
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
                            "query_budget_source": "CONFIG_250"
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
            "runtime": wall_clock
        })

    def write_csv(name, data):
        if not data: return
        all_keys = set()
        for d in data: all_keys.update(d.keys())
        all_keys = list(data[0].keys()) + [k for k in all_keys if k not in data[0].keys()]
        for d in data:
            for k in all_keys:
                if k not in d: d[k] = ""
        with open(out_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(data)
            
    overlap_data = []
    for seed, round_dict in method_cands.items():
        for round_idx, methods_dict in round_dict.items():
            mlist = list(methods_dict.keys())
            for i in range(len(mlist)):
                for j in range(i+1, len(mlist)):
                    mA, mB = mlist[i], mlist[j]
                    setA, setB = methods_dict[mA], methods_dict[mB]
                    inter = len(setA.intersection(setB))
                    uni = len(setA.union(setB))
                    jaccard = inter / uni if uni > 0 else 0.0
                    
                    hsetA = method_hard_cands[seed][round_idx].get(mA, set())
                    hsetB = method_hard_cands[seed][round_idx].get(mB, set())
                    hinter = len(hsetA.intersection(hsetB))
                    huni = len(hsetA.union(hsetB))
                    hjaccard = hinter / huni if huni > 0 else 0.0
                    
                    overlap_data.append({
                        "seed": seed, "round": round_idx,
                        "method_a": mA, "method_b": mB,
                        "intersection_count": inter,
                        "union_count": uni,
                        "jaccard_overlap": jaccard,
                        "hard_intersection_count": hinter,
                        "hard_union_count": huni,
                        "hard_jaccard_overlap": hjaccard
                    })

    val_eff_data = []
    for seed, round_dict in val_eff.items():
        for round_idx, methods_dict in round_dict.items():
            smc_data = methods_dict.get("smc")
            vsmc_data = methods_dict.get("validity_smc")
            if smc_data:
                smc_row = {"method": "smc", "seed": seed, "round": round_idx}
                smc_row.update(smc_data)
                val_eff_data.append(smc_row)
            if vsmc_data:
                vsmc_row = {"method": "validity_smc", "seed": seed, "round": round_idx}
                vsmc_row.update(vsmc_data)
                val_eff_data.append(vsmc_row)
            if smc_data and vsmc_data:
                diff_row = {"method": "Validity-SMC minus SMC", "seed": seed, "round": round_idx}
                for k in ["validity_rate", "valid_per_target_query", "mean_drift", "drift_per_target_query", "ess_ratio", "unique_candidates"]:
                    diff_row[k] = vsmc_data[k] - smc_data[k]
                val_eff_data.append(diff_row)

    write_csv("phase2_rounds.csv", rounds_data)
    write_csv("phase2_attackers.csv", attackers_data)
    write_csv("phase2_cem_collapse.csv", dynamics_data["cem"])
    write_csv("phase2_smc_degeneracy.csv", dynamics_data["smc"] + dynamics_data["validity_smc"])
    write_csv("phase2_defenders.csv", defenders_data)
    write_csv("phase2_transfer.csv", transfer_data)
    write_csv("phase2_redistribution.csv", redistribution_data)
    write_csv("phase2_clean_robust_tradeoff.csv", tradeoff_data)
    write_csv("phase2_diversity.csv", diversity_data)
    write_csv("phase2_summary.csv", summary_data)
    write_csv("phase2_g_adapt.csv", g_adapt_raw)
    write_csv("phase2_attacker_overlap.csv", overlap_data)
    write_csv("phase2_validity_efficiency.csv", val_eff_data)
    
    stats = []
    def calc_stats(metric_name, data, value_key, method_key="method"):
        method_groups = defaultdict(list)
        for d in data:
            val = optional_float(d.get(value_key))
            if not np.isnan(val): method_groups[d[method_key]].append(val)
                
        for meth, vals in method_groups.items():
            stats.append({
                "metric": metric_name, "method": meth, "round": "all",
                "n": len(vals), "mean": np.mean(vals), "sample_std": np.std(vals, ddof=1) if len(vals)>1 else 0,
                "median": np.median(vals), "min": np.min(vals), "max": np.max(vals),
                "positive_fraction": sum(1 for v in vals if v > 0) / len(vals) if len(vals) > 0 else np.nan
            })
            
    calc_stats("mean_drift", attackers_data, "mean_drift")
    calc_stats("G_adapt", g_adapt_raw, "G_adapt")
    calc_stats("defender_gain_difference", defenders_data, "defender_gain_difference")
    calc_stats("T_Evo", transfer_data, "T_Evo")
    calc_stats("ESS_ratio", dynamics_data["smc"] + dynamics_data["validity_smc"], "ess_ratio")
    calc_stats("normalized_entropy", dynamics_data["cem"], "normalized_entropy")
    write_csv("phase2_statistics.csv", stats)

    print("Aggregation complete")

if __name__ == "__main__": 
    open("docs/ARENA_PHASE2_LABEL_MAPPING.md", "w").write("# Phase 2 Active Attacker Exact Label Mapping\n")
    aggregate()
