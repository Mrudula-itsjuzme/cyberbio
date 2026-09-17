# Functional Group Replacement Library

| motif_id | source pattern | replacement | attachment rule | allowed context | validation rule | known limitations |
|---|---|---|---|---|---|---|
| OH_replace | [*]-C | [*]-O | single bond | aliphatic C | valency check | Does not support aromatic substitution |
| NH2_replace | [*]-C | [*]-N | single bond | aliphatic C | valency check | Does not support aromatic substitution |
| CF3_replace | [*]-C | [*]-C(F)(F)F | single bond | aliphatic C | valency check | Steric clash not checked |
| CN_replace | [*]-C | [*]-C#N | single bond | aliphatic C | valency check | N/A |
| F_replace | [*]-C | [*]-F | single bond | aliphatic C | valency check | N/A |
