import re
with open("/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/src/attackers/evolutionary.py", "r") as f:
    text = f.read()

repl = """        from materials_adv.domain.chemistry.plausibility import compute_tanimoto_similarity
        from src.metrics.distance import levenshtein
        try:
            tan_sim = compute_tanimoto_similarity(source_sequence, best_candidate)
        except:
            tan_sim = 0.0
        edit_dist = levenshtein(source_sequence, best_candidate)

        res = AttackResult(
            source_id=source_id,
            source_sequence=source_sequence,
            candidate_sequence=best_candidate,
            attack_name=self.name,
            attack_family=self.family,
            seed=rng.bit_generator.state["state"]["state"] if hasattr(rng, "bit_generator") else 0, 
            query_count=budget.queries_used,
            generation_count=budget.generations_used,
            runtime_seconds=budget.runtime_seconds,
            valid_rdkit=validator.is_valid_rdkit(best_candidate),
            constraint_pass=validator.is_valid_plausible(best_candidate),
            tanimoto_similarity=tan_sim if tan_sim is not None else 0.0, 
            edit_distance=edit_dist, 
            source_prediction=source_pred,
            candidate_prediction=best_pred,
            prediction_drift=best_drift,"""
text = re.sub(r'        res = AttackResult\(.*prediction_drift=best_drift,', repl, text, flags=re.DOTALL)
with open("/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/src/attackers/evolutionary.py", "w") as f:
    f.write(text)
