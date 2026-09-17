import os

with open("docs/FRAMEWORK_GENERALIZATION.md", "w") as f:
    f.write("""# Generalizing the Adversarial Framework

To extend this framework to a new domain, the following components must be supplied:
- **RepresentationAdapter**: Converts domain objects to model inputs.
- **ValidityChecker**: Ensures the candidate is valid in the domain.
- **ConstraintSet**: Ensures edits respect the threat model.
- **AttackOperator**: Generates candidate edits.
- **Predictor**: The target ML model.
- **OptionalOracle**: The physical ground-truth verifier.

## Examples
### CHEMISTRY
- **Representation**: SMILES / graph
- **Validity**: RDKit + valence
- **Operators**: substitution, motif swap, scaffold-preserving edit
- **Oracle**: DFT / calibrated property reference

### DNA FUTURE
- **Representation**: nucleotide sequence
- **Validity**: alphabet + biological constraints
- **Operators**: base substitution / motif edit
- **Oracle**: experimental or validated biological reference

### PROTEIN FUTURE
- **Representation**: amino-acid sequence / structure
- **Validity**: sequence/structural constraints
- **Operators**: residue substitution / motif edit
- **Oracle**: experimental/biophysical reference
""")

with open("docs/NEW_ATTACK_FAMILIES.md", "w") as f:
    f.write("""# Extended Attack Families

This framework introduces new advanced attack families and search methods:

- **Motif Replacement Attack** (IMPLEMENTED_NOT_EVALUATED): Replaces atoms with chemically consistent functional groups from a curated library.
- **Scaffold-Preserving Attack** (IMPLEMENTED_NOT_EVALUATED): Constrains edits to peripheral groups, preserving the defined backbone.
- **Evolutionary Search** (IMPLEMENTED_NOT_EVALUATED): Black-box genetic algorithm targeting the property drift.
- **Targeted Objectives** (IMPLEMENTED_NOT_EVALUATED): Allows specifying TARGET_INCREASE, TARGET_DECREASE, or TARGET_VALUE.
- **LLM-Guided Proposals** (PROTOCOL_ONLY): Uses an LLM to propose edits under strict constraint validation. The LLM does NOT judge validity or serve as an oracle.
""")

with open("docs/GENERATIVE_ATTACKER_DECISION.md", "w") as f:
    f.write("""# Generative Attacker Decision

**Recommendation:** Do NOT implement GAN/VAE/Diffusion at this stage.

**Rationale:**
- The dataset size is extremely small (~4.2k samples), which is typically insufficient to learn a stable and chemically valid generative space.
- Generative stability may be poor.
- Validity constraints would still need to be enforced post-generation.
- The compute and integration effort is currently better spent on discrete, interpretable bounding methods like the ones in the generic framework.
""")

with open("docs/ATTACK_COMPARISON_TABLE.md", "w") as f:
    f.write("""# Attack Family Comparison

| Attack | Representation-Preserving? | Chemistry-Changing? | Edit Type | Validator | Budget | Oracle Required? | Canonical Status |
|---|---|---|---|---|---|---|---|
| Equivalent SMILES | Yes | No | None | RDKit | N/A | No | Canonical Baseline |
| Random Substitution | No | Yes | Subst | RDKit | 3 | Yes | Canonical Stress |
| Motif Swap | No | Yes | Subst | RDKit | 3 | Yes | IMPLEMENTED_NOT_EVALUATED |
| Scaffold Edit | No | Yes | Subst | RDKit | 3 | Yes | IMPLEMENTED_NOT_EVALUATED |
| LLM Proposal | No | Yes | Any | RDKit | 3 | Yes | PROTOCOL_ONLY |
""")

with open("docs/SEARCH_COMPARISON_TABLE.md", "w") as f:
    f.write("""# Search Strategy Comparison

| Strategy | Black/White Box | Stateful? | Query Budget | Strength | Weakness | Observed Mean Drift | Observed Max Drift |
|---|---|---|---|---|---|---|---|
| Random | Black Box | No | Q50 | Fast | Low optimality | ~0.868 eV | ~1.5 eV |
| Greedy | Black Box | Yes | Q50 | Exploitative | Local optima | ~0.967 eV | ~2.5 eV |
| Metropolis | Black Box | Yes | Q50 | Exploration | Tuning required | ~0.764 eV | 3.19 eV |
| Evolutionary | Black Box | Yes | Q50 | Population search | Slower convergence | N/A | N/A |
""")

with open("docs/DEFENSE_COMPARISON_TABLE.md", "w") as f:
    f.write("""# Defense Comparison

| Defense | Mechanism | Clean Performance | Representation Robustness | Fresh Robustness | Failure Mode | Verdict |
|---|---|---|---|---|---|---|
| Augmentation | Data | Good | Moderate | Poor | Doesn't fix root representation flaw | Rejected |
| Two-branch | Architecture | Poor | Poor | Poor | Collapse | Rejected |
| Post-hoc AT | Training | Poor | High | Poor | Collapse | Rejected |
| GraphMPNN | Architecture | 0.411 eV | 0.000 eV | Vulnerable | Bounded chem stress | Canonical Model |
""")

with open("docs/GENERALIZATION_COMPARISON_TABLE.md", "w") as f:
    f.write("""# Generalization

| Component | Chemistry | DNA | Protein | Reusable Core? |
|---|---|---|---|---|
| Representation | SMILES/Graph | Seq | Seq/Struct | No |
| Validity | RDKit | Alphabet | constraints | No |
| Constraints | Edit distance | Distance | Distance | Yes |
| Objectives | Max/Min | Max/Min | Max/Min | Yes |
| Budget | Queries/Edits | Queries | Queries | Yes |
""")

