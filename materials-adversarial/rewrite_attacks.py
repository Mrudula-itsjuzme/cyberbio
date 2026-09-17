import os

with open("docs/FUNCTIONAL_GROUP_LIBRARY.md", "w") as f:
    f.write("""# Functional Group Replacement Library

| motif_id | source pattern | replacement | attachment rule | allowed context | validation rule | known limitations |
|---|---|---|---|---|---|---|
| OH_replace | [*]-C | [*]-O | single bond | aliphatic C | valency check | Does not support aromatic substitution |
| NH2_replace | [*]-C | [*]-N | single bond | aliphatic C | valency check | Does not support aromatic substitution |
| CF3_replace | [*]-C | [*]-C(F)(F)F | single bond | aliphatic C | valency check | Steric clash not checked |
| CN_replace | [*]-C | [*]-C#N | single bond | aliphatic C | valency check | N/A |
| F_replace | [*]-C | [*]-F | single bond | aliphatic C | valency check | N/A |
""")

os.makedirs("src/materials_adv/domain/chemistry/attacks", exist_ok=True)

with open("src/materials_adv/domain/chemistry/attacks/functional_group.py", "w") as f:
    f.write("""from typing import List
from rdkit import Chem
from materials_adv.framework.interfaces import AttackOperator, Candidate
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

class FunctionalGroupReplacementAttack(AttackOperator):
    \"\"\"
    Replaces a single atom (or fragment) with a motif from a predefined library.
    Edit: Atom substitution.
    Validity: RDKit valence.
    Attachment: Simple single bond replacement.
    \"\"\"
    def __init__(self, motifs=None):
        # Format: (smarts_pattern, replacement_smiles)
        self.motifs = motifs or [
            ("[C;X4;h1,h2,h3]", "O"), # Replace aliphatic C with OH
            ("[C;X4;h1,h2,h3]", "N"), # Replace aliphatic C with NH2
            ("[C;X4;h1,h2,h3]", "F"),
            ("[C;X4;h1,h2,h3]", "C#N"),
            ("[C;X4;h1,h2,h3]", "C(F)(F)F")
        ]
        self.validator = RDKitValidityChecker()

    def apply(self, candidate: Candidate) -> List[Candidate]:
        candidates = []
        mol = Chem.MolFromSmiles(candidate.identifier)
        if not mol: return []
        
        # NOTE: Full reliable functional group replacement in RDKit requires ReplaceSubstructs
        # or reaction SMARTS. For this Framework V2 evaluation, we use a simplified
        # atom-replacement strategy that mutates the SMILES or uses basic RDKit replacer.
        # This implementation mutates an atom to a functional group, effectively expanding the graph.
        
        for pattern_smarts, repl_smiles in self.motifs:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            repl = Chem.MolFromSmiles(repl_smiles)
            if not pattern or not repl: continue
            
            # Using ReplaceSubstructs
            try:
                # ReplaceSubstructs can generate multiple products depending on match
                products = Chem.ReplaceSubstructs(mol, pattern, repl, replaceAll=False)
                for p in products:
                    try:
                        Chem.SanitizeMol(p)
                        smi = Chem.MolToSmiles(p)
                        c = Candidate(identifier=smi, provenance=candidate.provenance + ["FunctionalGroup"])
                        if self.validator.is_valid(c):
                            candidates.append(c)
                    except Exception:
                        pass
            except Exception:
                pass
                
        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.identifier not in seen and c.identifier != candidate.identifier:
                seen.add(c.identifier)
                unique_candidates.append(c)
                
        return unique_candidates
""")

with open("src/materials_adv/domain/chemistry/attacks/scaffold_preserving.py", "w") as f:
    f.write("""from typing import List
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from materials_adv.framework.interfaces import AttackOperator, Candidate
from materials_adv.domain.chemistry.validator import RDKitValidityChecker

class ScaffoldPreservingAttack(AttackOperator):
    \"\"\"
    Modifies peripheral groups while strictly preserving the Murcko scaffold.
    Edit: Atom/fragment substitution off-scaffold.
    Validity: RDKit + Murcko scaffold identity check.
    \"\"\"
    def __init__(self, core_scaffold_smiles: str, base_operator: AttackOperator):
        self.core_scaffold_smiles = core_scaffold_smiles
        self.base_operator = base_operator
        self.validator = RDKitValidityChecker()
        
    def _get_scaffold(self, smi: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol: return ""
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return ""

    def apply(self, candidate: Candidate) -> List[Candidate]:
        candidates = []
        
        # Apply base operator (e.g. random substitution or functional group)
        proposed = self.base_operator.apply(candidate)
        
        for c in proposed:
            if self.validator.is_valid(c):
                # Check scaffold preservation
                c_scaffold = self._get_scaffold(c.identifier)
                if c_scaffold == self.core_scaffold_smiles:
                    c.provenance.append("ScaffoldPreserving")
                    candidates.append(c)
                    
        return candidates
""")

