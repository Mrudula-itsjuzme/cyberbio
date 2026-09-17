# Attack Family Comparison

Generated from `results/framework_v2/verified_run_1/`.

| Attack | Representation-preserving? | Chemistry-changing? | Edit unit | Validator | Budget | Oracle required? | Status |
|---|---|---|---|---|---|---|---|
| Equivalent SMILES | yes | no | none | RDKit canonical | n/a | no | canonical baseline |
| Random substitution | no | yes | 1 atom per application | RDKit + connectivity + attachment | 3 | no | canonical stress |
| SimpleSubstitution (V2) | no | yes | 1 atom per application | RDKit + connectivity + attachment | 3 | no | VERIFIED |
| AliphaticCarbonSubstitutionAttack | no | yes | 1 application = up to 4 atom edits | RDKit + connectivity + attachment | 3 | no | VERIFIED |
| Scaffold-preserving | no | yes | inherited from base operator | RDKit + Murcko subgraph retention | 3 | no | VERIFIED (subset) |
| True fragment/subgraph replacement | no | yes | fragment swap | not implemented | n/a | no | DESIGN ONLY |

Operator comparison (fixed strategy: greedy; from `operator_summary.csv`):

| Operator | Q | Mean drift (eV) | Max (eV) | Mean operator edits | Max operator edits | Max atom edits | Max structural-change proxy | Drift per query (eV) | Runs with no accepted edit |
|---|---|---|---|---|---|---|---|---|---|
| aliphatic_carbon_substitution | 10 | 0.425 | 1.289 | 1.185 | 2 | 2 | 2 | 0.059 | 3 |
| aliphatic_carbon_substitution | 20 | 0.569 | 1.781 | 1.667 | 3 | 5 | 8 | 0.051 | 3 |
| aliphatic_carbon_substitution | 50 | 0.760 | 2.591 | 2.556 | 3 | 10 | 16 | 0.047 | 3 |
| scaffold_preserving | 10 | 0.067 | 0.526 | 1.000 | 1 | 1 | 1 | 0.007 | 25 |
| scaffold_preserving | 20 | 0.068 | 0.526 | 1.000 | 1 | 1 | 1 | 0.003 | 25 |
| scaffold_preserving | 50 | 0.139 | 1.291 | 2.200 | 3 | 3 | 3 | 0.003 | 25 |
| simple_substitution | 10 | 0.947 | 1.894 | 1.000 | 1 | 1 | 1 | 0.095 | 0 |
| simple_substitution | 20 | 1.188 | 2.495 | 1.200 | 3 | 3 | 3 | 0.059 | 0 |
| simple_substitution | 50 | 1.703 | 3.055 | 2.133 | 3 | 3 | 3 | 0.038 | 0 |

`AliphaticCarbonSubstitutionAttack` was previously misnamed
`FunctionalGroupReplacementAttack`; it substitutes aliphatic carbons and does not
manipulate functional-group fragments (audit section 10). Its operator-edit count is not
comparable to its atom-edit count, so both are reported.

Framework-V2 attack numbers are `DEVELOPMENTAL_UNVERIFIED` for the developmental run and
`VERIFIED` only for `verified_run_1`.
