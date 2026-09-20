import sys
import json
import os
import hashlib
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

sys.path.append('src')
from materials_adv.domain.chemistry.validator import RDKitValidityChecker
from materials_adv.framework.interfaces import Candidate
from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.substitution import SubstitutionAttack
from materials_adv.attacks.insertion import InsertionAttack
from materials_adv.attacks.rearrangement import RearrangementAttack
from materials_adv.attacks.search.proposals import CompositeProposalOperator
from materials_adv.data.tokenizer import tokenize

# Load canonical vocab
with open('data/processed/vocab.json', 'r') as f:
    vocab = json.load(f)

# Canonical validator
canonical_validator = RDKitValidityChecker()

def get_proposer(seed):
    rng = np.random.default_rng(seed)
    ops = [
        DeletionAttack(rng=rng, attack_budget=1),
        SubstitutionAttack(rng=rng, allowed_tokens=vocab, attack_budget=1),
        InsertionAttack(rng=rng, allowed_tokens=vocab, attack_budget=1),
        RearrangementAttack(rng=rng, window_size=3)
    ]
    return CompositeProposalOperator(operators=ops, rng=rng)

def get_deterministic_seed(global_seed, candidate_id, replication):
    s = f"{global_seed}_{candidate_id}_{replication}"
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest()[:8], 16)

def estimate_local_boundary(candidate_smiles, k=10, global_seed=42, candidate_id="unknown", replication=0):
    seed = get_deterministic_seed(global_seed, candidate_id, replication)
    local_rng = np.random.default_rng(seed)
    proposer = get_proposer(seed)
    
    tokens = tokenize(candidate_smiles, on_unknown="char")
    
    # We will enumerate proposals until we hit enough, or try propose multiple times.
    # CompositeProposalOperator has `propose(tokens)` which returns a Proposal.
    # Let's generate max 50 proposals to find up to k unique
    proposals_dict = {} # ordered dict implicitly in python 3.7+
    for _ in range(50):
        p = proposer.propose(tokens)
        if p is not None:
            new_smi = "".join(p.tokens)
            if new_smi not in proposals_dict:
                proposals_dict[new_smi] = p
                
    unique_proposals = list(proposals_dict.keys())
    
    # Deterministic shuffle
    unique_proposals.sort() # sort to ensure order-independence before shuffle
    local_rng.shuffle(unique_proposals)
    
    chosen = unique_proposals[:k]
    
    if not chosen:
        return 0.0, 0.0, 0, 0, 0, 0, 0
    
    valid_count = 0
    for new_smiles in chosen:
        cand = Candidate(identifier=new_smiles)
        if canonical_validator.is_valid(cand):
            valid_count += 1
            
    evaluated = len(chosen)
    p_valid_local = valid_count / float(evaluated)
    u_valid = 4 * p_valid_local * (1 - p_valid_local)
    
    return p_valid_local, u_valid, k, len(unique_proposals), evaluated, valid_count, evaluated - valid_count

if __name__ == '__main__':
    MIN_NEIGHBORS = 5
    
    os.makedirs('results/validity_boundary', exist_ok=True)
    c_data = []
    
    with open('results/phase2_paired_benchmark/clean_attack_records.jsonl') as f:
        lines = f.readlines()
        
    parsed_lines = []
    for line in lines:
        try:
            parsed_lines.append(json.loads(line))
        except:
            pass
            
    df_all = pd.DataFrame(parsed_lines)
    
    method_counts = []
    
    sampled_indices = []
    methods_present = df_all.get('attack_type', pd.Series([])).dropna().unique()
    for m in methods_present:
        sub = df_all[df_all['attack_type'] == m]
        sampled = sub.sample(n=min(len(sub), 200), random_state=42)
        sampled_indices.extend(sampled.index)
        method_counts.append({
            'method': m,
            'available_candidates': len(sub),
            'sampled_candidates': len(sampled),
            'unique_sources': sampled['sample_id'].nunique()
        })
        
    sample_recs = df_all.loc[sampled_indices].to_dict('records')
    
    global_seed = 2026
    
    valid_candidates_by_method = {m: 0 for m in methods_present}
    excluded_candidates_by_method = {m: 0 for m in methods_present}
    
    for rec in sample_recs:
        attack_id = rec.get('attack_id')
        method = rec.get('attack_type')
        if method not in excluded_candidates_by_method:
            excluded_candidates_by_method[method] = 0
            valid_candidates_by_method[method] = 0
            
        adv_smiles = rec.get('adversarial_representation')
        drift = rec.get('absolute_prediction_drift')
        
        if pd.isna(drift) or not adv_smiles or pd.isna(attack_id):
            excluded_candidates_by_method[method] += 1
            continue
            
        # Run local probe 5 times to check stability
        u_vals = []
        evals_list = []
        for rep in range(5):
            p_val, u_val, req_k, uniq_p, evals, v_count, iv_count = estimate_local_boundary(
                adv_smiles, k=10, global_seed=global_seed, candidate_id=attack_id, replication=rep
            )
            u_vals.append(u_val)
            evals_list.append(evals)
            
        mean_evals = np.mean(evals_list)
        if mean_evals < MIN_NEIGHBORS:
            excluded_candidates_by_method[method] += 1
            continue
            
        valid_candidates_by_method[method] += 1
            
        c_data.append({
            'run_id': 'phase2_paired_benchmark',
            'source_id': rec['sample_id'],
            'candidate_id': attack_id,
            'method': method,
            'validity_status': rec['validity_status'],
            'drift': float(drift),
            'u_valid': np.mean(u_vals),
            'u_valid_std': np.std(u_vals),
            'evaluated_neighbors': mean_evals
        })

    df = pd.DataFrame(c_data)
    df.to_csv('results/validity_boundary/candidate_boundary_dataset.csv', index=False)
    
    # Fill in valid and excluded counts
    for m in method_counts:
        m['valid_candidates'] = valid_candidates_by_method.get(m['method'], 0)
        m['excluded_candidates'] = excluded_candidates_by_method.get(m['method'], 0)
    
    pd.DataFrame(method_counts).to_csv('results/validity_boundary/canonical_cohort_counts.csv', index=False)

    df_valid = df[df['validity_status'] == 'valid']
    
    # 10. Source-cluster bootstrap
    def source_cluster_bootstrap(df_subset, n_boot=1000):
        sources = df_subset['source_id'].unique()
        rhos = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            boot_sources = rng.choice(sources, size=len(sources), replace=True)
            # reconstructing dataframe from boot_sources
            boot_df = pd.concat([df_subset[df_subset['source_id'] == s] for s in boot_sources])
            r, _ = scipy.stats.spearmanr(boot_df['u_valid'], boot_df['drift'])
            rhos.append(r)
        return np.nanpercentile(rhos, 2.5), np.nanpercentile(rhos, 97.5)

    pooled_rho, pooled_pval = scipy.stats.spearmanr(df_valid['u_valid'], df_valid['drift'])
    ci_lower, ci_upper = source_cluster_bootstrap(df_valid, n_boot=500)
    
    print(f"Pooled Spearman: {pooled_rho:.3f} CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # 11. Within-source check
    # Center drift and u_valid within source
    df_valid.loc[:, 'drift_centered'] = df_valid.groupby('source_id')['drift'].transform(lambda x: x - x.mean())
    df_valid.loc[:, 'u_valid_centered'] = df_valid.groupby('source_id')['u_valid'].transform(lambda x: x - x.mean())
    
    within_rho, within_pval = scipy.stats.spearmanr(df_valid['u_valid_centered'].dropna(), df_valid['drift_centered'].dropna())
    print(f"Within-Source Spearman: {within_rho:.3f} p={within_pval:.3e}")
    
    # 13. Top-drift statistics
    top_10_thresh = df_valid['drift'].quantile(0.90)
    top_5_thresh = df_valid['drift'].quantile(0.95)
    top_10 = df_valid[df_valid['drift'] >= top_10_thresh]
    top_5 = df_valid[df_valid['drift'] >= top_5_thresh]
    
    def report_stats(name, d):
        print(f"{name}: n={len(d)}, sources={d['source_id'].nunique()}, mean_u={d['u_valid'].mean():.3f}, median_u={d['u_valid'].median():.3f}, std_u={d['u_valid'].std():.3f}")
        
    report_stats("ALL VALID", df_valid)
    report_stats("TOP 10%", top_10)
    report_stats("TOP 5%", top_5)
    
    # 12. Method composition check
    print("Method composition (ALL):", df_valid['method'].value_counts(normalize=True).to_dict())
    print("Method composition (TOP 10%):", top_10['method'].value_counts(normalize=True).to_dict())
    print("Method composition (TOP 5%):", top_5['method'].value_counts(normalize=True).to_dict())
