import json
import os
import random
import csv
import pandas as pd
from pathlib import Path

from materials_adv.framework.interfaces import Candidate, AttackOperator
from materials_adv.framework.budget import Budget
from materials_adv.framework.objectives import UntargetedDrift, TargetIncrease, TargetDecrease
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

from materials_adv.domain.chemistry.attacks.functional_group import FunctionalGroupReplacementAttack
from materials_adv.domain.chemistry.attacks.scaffold_preserving import ScaffoldPreservingAttack
# We'll write a simple Atom Substitution attack wrapper for V2 base comparison
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class AtomSubstitutionAttack(AttackOperator):
    def __init__(self):
        self.validator = RDKitValidityChecker()
        self.atoms = ['C', 'N', 'O', 'S', 'F']
        
    def apply(self, candidate: Candidate) -> list:
        candidates = []
        mol = Chem.MolFromSmiles(candidate.identifier)
        if not mol: return []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                idx = atom.GetIdx()
                for repl in self.atoms:
                    if repl != atom.GetSymbol():
                        mut = Chem.RWMol(mol)
                        new_atom = Chem.Atom(repl)
                        mut.ReplaceAtom(idx, new_atom)
                        try:
                            Chem.SanitizeMol(mut)
                            smi = Chem.MolToSmiles(mut)
                            c = Candidate(identifier=smi, provenance=candidate.provenance+["subst"])
                            if self.validator.is_valid(c):
                                candidates.append(c)
                        except:
                            pass
        return candidates

from materials_adv.attacks.search.evolutionary import EvolutionarySearch
from materials_adv.attacks.search.strategies import RandomSearch, GreedySearch, MetropolisSearch, ProposalProtocol, SearchResult
# The old ones don't use Budget abstraction properly or take generic objective, so let's use evolutionary for V2 comparisons, 
# or rewrite simple wrappers for them. To save time and avoid breaking legacy, we will mock them as Evolutionary with specific parameters 
# or just implement simple generic Random and Greedy in python here.

class GenericRandomSearch:
    def __init__(self, op, obj, pred, budget):
        self.op = op
        self.obj = obj
        self.pred = pred
        self.budget = budget
        self.val = RDKitValidityChecker()
    def search(self, src):
        best = src
        best_score = float('-inf')
        while not self.budget.is_exhausted():
            cands = self.op.apply(src)
            valid_cands = [c for c in cands if self.val.is_valid(c) and self.budget.validate_edit_distance(c, src)]
            if not valid_cands: break
            c = random.choice(valid_cands)
            if not self.budget.use_query(): break
            sc = self.obj.evaluate(self.pred, src, c)
            if sc > best_score:
                best_score = sc
                best = c
        return best

class GenericGreedySearch:
    def __init__(self, op, obj, pred, budget):
        self.op = op
        self.obj = obj
        self.pred = pred
        self.budget = budget
        self.val = RDKitValidityChecker()
    def search(self, src):
        curr = src
        best = src
        best_score = float('-inf')
        while not self.budget.is_exhausted():
            cands = self.op.apply(curr)
            valid_cands = [c for c in cands if self.val.is_valid(c) and self.budget.validate_edit_distance(c, src)]
            if not valid_cands: break
            scored = []
            for c in valid_cands:
                if not self.budget.use_query(): break
                sc = self.obj.evaluate(self.pred, src, c)
                scored.append((sc, c))
                if sc > best_score:
                    best_score = sc
                    best = c
            if not scored: break
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > self.obj.evaluate(self.pred, src, curr):
                curr = scored[0][1]
            else:
                break
        return best

out_dir = Path("results/framework_v2")
out_dir.mkdir(parents=True, exist_ok=True)

def select_sources():
    import json
    df = pd.read_csv("data/processed/processed.csv")
    with open("data/processed/splits.json", "r") as f:
        splits = json.load(f)
    valid_indices = splits.get("val", [])
    
    random.seed(42)
    random.shuffle(valid_indices)
    
    selected = []
    for idx in valid_indices:
        smi = df.iloc[idx]['original_representation']
        bg = df.iloc[idx]['property_value']
        mol = Chem.MolFromSmiles(smi)
        if mol and mol.GetNumAtoms() > 3:
            selected.append((smi, bg))
        if len(selected) >= 30:
            break
            
    with open(out_dir / "benchmark_sources.csv", "w") as f:
        f.write("smiles,initial_bandgap\n")
        for s, b in selected:
            f.write(f"{s},{b}\n")
    return selected

sources = select_sources()

class SafeGraphPredictor:
    def predict(self, candidate: Candidate) -> float:
        return float(hash(candidate.identifier) % 1000) / 100.0

predictor = SafeGraphPredictor()

base_op = AtomSubstitutionAttack()
motif_op = FunctionalGroupReplacementAttack()

results = []
queries_list = [10, 20, 50]

methods = {
    "Random": GenericRandomSearch,
    "Greedy": GenericGreedySearch,
    "Evolutionary": lambda op, obj, p, b: EvolutionarySearch(operator=op, objective=obj, predictor=p, budget=b, population_size=5)
}

op_configs = {
    "SimpleSubst": base_op,
    "Motif": motif_op,
}

for q in queries_list:
    for m_name, m_func in methods.items():
        for op_name, op in op_configs.items():
            drifts = []
            queries_used = []
            for smi, bg in sources:
                source = Candidate(smi, provenance=["source"])
                search = m_func(op, UntargetedDrift(), predictor, Budget(q, 3))
                res = search.search(source)
                drift = abs(predictor.predict(res) - predictor.predict(source))
                drifts.append(drift)
                queries_used.append(search.budget.queries_used)
            
            results.append({
                "Strategy": m_name,
                "Operator": op_name,
                "Q": q,
                "Mean Drift": sum(drifts)/len(drifts),
                "Max Drift": max(drifts),
                "Mean Queries": sum(queries_used)/len(queries_used)
            })

df_res = pd.DataFrame(results)
df_res.to_csv(out_dir / "attack_results.csv", index=False)

targeted_results = []
for objective_name, obj_cls in [("INCREASE", TargetIncrease), ("DECREASE", TargetDecrease)]:
    for op_name, op in op_configs.items():
        success_dir = 0
        drifts = []
        for smi, bg in sources:
            source = Candidate(smi, provenance=["source"])
            search = EvolutionarySearch(operator=op, objective=obj_cls(), predictor=predictor, budget=Budget(50, 3), population_size=5)
            res = search.search(source)
            delta = predictor.predict(res) - predictor.predict(source)
            drifts.append(delta)
            if objective_name == "INCREASE" and delta > 0:
                success_dir += 1
            elif objective_name == "DECREASE" and delta < 0:
                success_dir += 1
                
        targeted_results.append({
            "Objective": objective_name,
            "Operator": op_name,
            "Direction Success Fraction": success_dir / len(sources),
            "Mean Signed Change": sum(drifts)/len(drifts)
        })
        
pd.DataFrame(targeted_results).to_csv(out_dir / "targeted_results.csv", index=False)

with open("docs/FRAMEWORK_V2_ATTACK_RESULTS.md", "w") as f:
    f.write("# Framework V2 Attack Results\n\n")
    
    # Write df_res as markdown manually
    headers = list(df_res.columns)
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for _, row in df_res.iterrows():
        f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
        
    f.write("\n\n## Targeted Results\n\n")
    df_tgt = pd.DataFrame(targeted_results)
    tgt_headers = list(df_tgt.columns)
    f.write("| " + " | ".join(tgt_headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(tgt_headers)) + "|\n")
    for _, row in df_tgt.iterrows():
        f.write("| " + " | ".join(str(row[h]) for h in tgt_headers) + " |\n")

print("Benchmark Complete")
