# Research Questions

Evidence paths refer to `results/final/attack_strategy_comparison.csv` unless stated otherwise.

| RQ | Status | Answer | Evidence | Limitation / next step |
|---|---|---|---|---|
| 1 Transformer polymer representation sensitivity? | ANSWERED ELSEWHERE | Evidence indicates sensitivity, not physical error. | Canonical artifacts | Revalidate with physical oracle. |
| 2 Randomized representations expose instability? | ANSWERED ELSEWHERE | Yes descriptively. | RandomSMILES work | Separate equivalence from chemical edits. |
| 3 Constrained local search adds sensitivity? | EXECUTED | Yes, nonzero drift occurs. | Classical rows | Single seed/N=30. |
| 4 MCMC beats random? | PARTIAL | Sometimes descriptively; inspect paired p-values. | `statistical_results.json` | No universal win. |
| 5 Evolutionary improves global exploration? | PARTIAL | Strong descriptive Materials results at high budget. | Budget-50 rows | Cost and power prevent promotion. |
| 6 Attribution improves efficiency? | ANSWERED | Not consistently after attribution queries. | Query columns | Fixed, nonadaptive attribution. |
| 7 High beats low/random? | PARTIAL | Often directionally, not universally/statistically. | Paired tests | Multiple comparisons exploratory. |
| 8 Learned generator improves validity? | NEGATIVE CROSS-DOMAIN | Bio is syntactically valid; Materials trained run accepted none. | Generator rows | Architecture/sampling may be defective. |
| 9 Trained beats untrained? | INCOMPLETE | Bio measurable; Materials control blocked. | Quality summary | Run true bounded Materials control. |
| 10 LLM useful proposals? | BLOCKED | Unknown. | Mock tests only | Real credentials/provider run. |
| 11 RL better adaptive policy? | BLOCKED | Unknown. | Protocol only | Implement and train with held-out test. |
| 12 GAN suitable? | REJECTED | Poor fit under current discrete constraints. | Feasibility analysis | Revisit only with justified discrete estimator. |
| 13 Drift/query comparison? | EXECUTED | Varies by domain and budget; no universal winner. | Final figures | Generator compute is separate. |
| 14 Cross-domain transfer? | NEGATIVE | No family has validated superiority in both domains. | Cross-domain table | Different validity/success semantics. |
| 15 Validity bottlenecks? | ANSWERED | Materials attribution-low and generator are notably constrained. | Validity rates | RDKit is only parse validity. |
| 16 Poor scaling? | ANSWERED | Attribution, evolutionary, generator, RL/LLM carry extra costs. | Runtime/query logs | Hardware-specific runtime. |
| 17 Representation vs chemistry attacks differ? | ANSWERED CONCEPTUALLY | Yes; only the former may preserve semantics. | Math/validity docs | Physical equivalence unverified. |
| 18 Defense transfer? | PARTIAL | Not universal; Materials not validated. | Existing defense artifacts | Run compatible checkpoints. |
| 19 Length associated with vulnerability? | PARTIAL ELSEWHERE | Association observed, causality not established. | Shortcut audit | Controlled ablation needed. |
| 20 Promote a method? | ANSWERED | None yet. | Status matrix | Require multiseed, fair advantage, cost, validity. |
