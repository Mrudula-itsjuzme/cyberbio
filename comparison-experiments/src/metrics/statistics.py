from typing import List, Dict, Any
import numpy as np

from src.core.schemas import AttackResult

def compute_comparison_statistics(results: List[AttackResult]) -> Dict[str, Any]:
    if not results:
        return {}
        
    drifts = [r.prediction_drift for r in results]
    shifts = [r.candidate_prediction - r.source_prediction for r in results]
    valid_rdkit = [r.valid_rdkit for r in results]
    valid_constraint = [r.constraint_pass for r in results]
    
    # Define success as finding a valid constraint-passing candidate with drift > 0.05
    # The canonical project often used specific criteria, we use drift threshold 0.1 for now, or configurable.
    successes = [1 if r.constraint_pass and r.prediction_drift > 0.1 else 0 for r in results]
    
    queries = [r.query_count for r in results]
    runtimes = [r.runtime_seconds for r in results]
    
    unique_cands = len(set(r.candidate_sequence for r in results if r.candidate_sequence != r.source_sequence))
    total_cands = sum(1 for r in results if r.candidate_sequence != r.source_sequence)
    unique_rate = unique_cands / total_cands if total_cands > 0 else 0.0
    
    return {
        "attacker": results[0].attack_name,
        "count": len(results),
        "mean_drift": float(np.mean(drifts)),
        "median_drift": float(np.median(drifts)),
        "max_drift": float(np.max(drifts)),
        "p90_drift": float(np.percentile(drifts, 90)),
        "success_rate": float(np.mean(successes)),
        "valid_rdkit_rate": float(np.mean(valid_rdkit)),
        "constraint_pass_rate": float(np.mean(valid_constraint)),
        "mean_tanimoto": float(np.mean([r.tanimoto_similarity for r in results])),
        "unique_candidate_rate": unique_rate,
        "mean_duplicate_proposals": float(np.mean([r.duplicate_proposals for r in results])),
        "mean_queries": float(np.mean(queries)),
        "total_queries": int(np.sum(queries)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
    }
